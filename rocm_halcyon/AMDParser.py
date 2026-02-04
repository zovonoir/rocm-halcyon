from ast import Tuple
from fileinput import filename
import os
import pandas as pd
import numpy as np
import json
import tqdm
from warnings import warn as warning
from typing import List 
from dataclasses import dataclass
from typing import Optional, Any,Dict
import parse
import re
from perfetto.trace_processor import TraceProcessor

# from utils import CpuOp
from .KernelDef import CpuOp,Kernel

class AMDTorchProfilerParser():
    def __init__(self,json_file):
        self.json_file = json_file
        self.all_kernels:List[Kernel] = []
        self.all_cpu_ops:List[CpuOp] = []
        self._gc_counter = 0

    def _filter_kernels(self,all_events):
        return [obj for obj in all_events if "cat" in obj and obj['cat']=="kernel"]

    def _filter_launch_kernels(self,all_events):
        cond_is_hipLaunchKernel = lambda obj:("cat" in obj and \
                                obj["cat"]=="cuda_runtime" and \
                                 "name" in obj and re.match(r'^hip[a-zA-Z]*LaunchKernel$', obj["name"]))

        cond_is_cuLaunchKernel = lambda obj:("cat" in obj and \
                                obj["cat"]=="cuda_driver" and \
                                 "name" in obj and re.match(r'^cu[a-zA-Z]*LaunchKernel[a-zA-Z]*$', obj["name"]))

        return [obj for obj in all_events if cond_is_hipLaunchKernel(obj)]
        # return [obj for obj in all_events if "cat" in obj and \
        #         obj["cat"]=="cuda_runtime" and \
        #         "name" in obj and re.match(r'hip[a-zA-Z]*LaunchKernel$', obj["name"])]

    def _filter_user_annotations(self,all_events):
        return [obj for obj in all_events if "cat" in obj and obj["cat"]=="gpu_user_annotation"]

    def _filter_cpu_op(self,all_events):
        return [obj for obj in all_events if "cat" in obj and obj["cat"]=="cpu_op"]
    
    def _find_cpu_op(self,start,end) -> CpuOp:
        candidate:List[CpuOp] = []
        for idx,item in enumerate(self.all_cpu_ops):
            if item.start_timestamp <= start and item.end_timestamp >= end:
                candidate.append(item)
            if item.start_timestamp > start:
                break
        
        if len(candidate) > 0:
            candidate.sort(key=lambda obj:obj.duration)
            res = candidate[0]
            self._gc_counter += 1
        else:
            res = None

        if self._gc_counter > 0 and self._gc_counter % 500 == 0:
            self._gc_counter = 0
            start_timestamp = res.start_timestamp * 0.999999999
            self.all_cpu_ops = [obj for obj in self.all_cpu_ops if obj.start_timestamp > start_timestamp]
    
        return res

    def structurize_kernel(self,all_kernels,correlation_to_kernel_launch):
        self.all_kernels:List[Kernel] = []

        for item in tqdm.tqdm(all_kernels,desc="scrutrizing all device kernels..."):
            smem = None
            host_launching_cost = None
            correlation = int(item["args"]["correlation"])
            if correlation in correlation_to_kernel_launch:
                smem = correlation_to_kernel_launch[int(item["args"]["correlation"])][0]["args"]["shared memory"]
                host_launching_cost = correlation_to_kernel_launch[int(item["args"]["correlation"])][0]["dur"]
            self.all_kernels.append(
                Kernel(
                    name = item["name"],
                    duration = item["dur"],
                    start_timestamp = item["ts"],
                    end_timestamp = item["ts"] + item["dur"],
                    device = item["args"]["device"],
                    stream = item["args"]["stream"],
                    correlation = correlation,
                    grid = item["args"]["grid"],
                    block = item["args"]["block"],
                    smem = smem,
                    host_launching_cost=host_launching_cost,
                    annotation = None, # will support in future,
                    gap = None,
                    input_dtype = None,
                    input_shape = None,
                    input_strides = None,
                    output_shape = None,
                    output_dtype = None,
                    output_strides = None,
                    cpu_op_name = None,
                    torch_op_name = None 
                )
            )
        self.all_kernels.sort(key=lambda obj: obj.start_timestamp)
        # 查找同一个device,同一个stream上的kernel gap
        # 这里需要加上在同一个stream内的限制条件
        self.all_kernels[0].gap = 0
        for i in range(1,len(self.all_kernels)):
            self.all_kernels[i].gap = self.all_kernels[i].start_timestamp - self.all_kernels[i-1].end_timestamp
    
    def structurize_cpu_op(self,all_cpu_ops):
        # breakpoint()
        self.all_cpu_ops:List[CpuOp] = []
        for cpu_op in tqdm.tqdm(all_cpu_ops,desc="structurizing alll cpu ops..."):
            input_shape = None if "Input Dims" not in cpu_op['args'] else cpu_op['args']["Input Dims"]
            input_strides = None if "Input Strides" not in cpu_op['args'] else cpu_op['args']["Input Strides"]
            input_dtype = None if "Input type" not in cpu_op['args'] else cpu_op['args']["Input type"]
            concrete_inputs = None if "Concrete Inputs" not in cpu_op['args'] else cpu_op['args']["Concrete Inputs"]
            if concrete_inputs is not None and input_shape is not None:
                s = []
                for scalar,tensor in zip(concrete_inputs,input_shape):
                    if tensor == []:
                        if scalar != '':
                            s.append(scalar)
                        else:
                            s.append("UNKNOWN")
                    else:
                        s.append(tensor)
                input_shape = s

            # 目前为止,我没有看到包含输出形状的torch profiler文件,或许torch profiler只能捕获输入形状
            # 输出形状需要另外写推理函数
            output_shape = None # if "Input Dims" not in cpu_op['args'] else cpu_op['args']["Input Dims"] 
            output_strides = None
            output_dtype = None

            self.all_cpu_ops.append(
                CpuOp(
                    name = cpu_op['name'],
                    start_timestamp = cpu_op['ts'],
                    end_timestamp = cpu_op['ts'] + cpu_op['dur'],
                    input_shape=input_shape,
                    output_shape=output_shape,
                    input_strides = input_strides,
                    output_strides = output_strides,
                    duration = cpu_op['dur'],
                    input_dtype = input_dtype,
                    output_dtype = output_dtype,
                )
            )

        self.all_cpu_ops.sort(key=lambda obj: obj.start_timestamp)

    def mapping_input_shapes(self,):
        # 对于每一条hiplaunchkernel信号,需要找到上层那个cpu_op,因为可能有形状信息
        for kernel in tqdm.tqdm(self.all_kernels,desc="mapping device kernel to cpu op..."):
            corr_id = kernel.correlation #kernel["args"]["correlation"]
            if corr_id in self.correlation_to_kernel_launch:
                launch_kernel_start_timestamp = self.correlation_to_kernel_launch[corr_id][0]["ts"]
                launch_kernel_end_timestamp = self.correlation_to_kernel_launch[corr_id][0]["ts"] + \
                                self.correlation_to_kernel_launch[corr_id][0]["dur"]
                cpu_op = self._find_cpu_op(launch_kernel_start_timestamp,launch_kernel_end_timestamp)
                
                if cpu_op is not None:
                    kernel.input_shape = cpu_op.input_shape
                    kernel.output_shape = cpu_op.output_shape
                    kernel.input_dtype = cpu_op.input_dtype
                    kernel.output_dtype = cpu_op.output_dtype
                    kernel.input_strides = cpu_op.input_strides
                    kernel.output_strides = cpu_op.output_strides
                    kernel.cpu_op_name = cpu_op.name
                else:
                    kernel.input_shape = None # cpu_op.input_shape
                    kernel.output_shape = None # cpu_op.output_shape
                    kernel.input_dtype = None # cpu_op.input_dtype
                    kernel.output_dtype = None # cpu_op.output_dtype
                    kernel.input_strides = None # cpu_op.input_strides
                    kernel.output_strides = None # cpu_op.output_strides
                    kernel.cpu_op_name = None # cpu_op.name

    def check_data_correctness(self,):
        if len(self.all_kernels) == 0:
            assert 0,"Provided json file didn't find any kernel information."
        if len(self.all_launch_kernels) == 0:
            warning("Provided json file didn't find any kernel launching information.")

    def parse(self):
        json_file = self.json_file
        with open(json_file, 'r') as file:
            data = json.load(file)
            all_events = data.get("traceEvents", [])
            if all_events == []:
                assert 0,"json file is empty."
            
            all_kernels = self._filter_kernels(all_events)
            all_launch_kernels = self._filter_launch_kernels(all_events)
            all_user_annotations = self._filter_user_annotations(all_events)
            all_cpu_op = self._filter_cpu_op(all_events)
            all_cpu_op.sort(key=lambda obj: obj["ts"])
            # 构建一个correlation id和hiplaunchkernel的对照表,可以根据kernel中的correlation id快速的找到下发信号
            correlation_to_kernel_launch = dict()

            for item in tqdm.tqdm(all_launch_kernels,desc="mapping kernel correlation..."):
                corr_id = item["args"]["correlation"]
                if corr_id not in correlation_to_kernel_launch:
                    correlation_to_kernel_launch[corr_id] = [item]
                else:
                    warning(f"correlation id:{corr_id} has repeated kernel launching signals!")

            self.all_launch_kernels = all_launch_kernels
            self.all_user_annotations = all_user_annotations
            self.correlation_to_kernel_launch = correlation_to_kernel_launch
            self.structurize_kernel(all_kernels,correlation_to_kernel_launch) # self.all_kernels
            self.structurize_cpu_op(all_cpu_op)
            self.mapping_input_shapes()
            # self.infer_output_shapes()

        self.check_data_correctness()
        return self.all_kernels
        # return self

    def export_to_excel_v2(self,visitors,file_name = None,sheet_name="kernel"):
        pass


    def export_to_excel(self,file_name = None,sheet_name="kernel"):
        raw_data = {
            "kernel_name":[],
            "cpu_op_name":[],
            "kernel_duration":[],
            "kernel_start_timestamp":[],
            "kernel_end_timestamp":[],
            "kernel_gap":[],
            "kernel_grid_size":[],
            "kernel_block_size":[],
            "kernel_stream":[],
            "kernel_shared_memory":[],
            "kernel_input_shape":[],
            "kernel_input_data_type":[],
            "kernel_output_shape":[],
            "kernel_output_data_type":[],
            "kernel_launching_cost":[],
            "kernel_running_device":[],
        }
        for kernel in tqdm.tqdm(self.all_kernels,desc="exporting..."):
            raw_data["kernel_name"].append(kernel.name)
            raw_data["kernel_duration"].append(kernel.duration)
            raw_data["kernel_start_timestamp"].append(str(kernel.start_timestamp))
            raw_data["kernel_end_timestamp"].append(str(kernel.end_timestamp))
            raw_data["kernel_block_size"].append(kernel.block)
            raw_data["kernel_grid_size"].append(kernel.grid)
            raw_data["kernel_shared_memory"].append(kernel.smem)
            raw_data["kernel_input_shape"].append(kernel.input_shape)
            raw_data["kernel_output_shape"].append(kernel.output_shape)
            raw_data["kernel_input_data_type"].append(kernel.input_dtype)
            raw_data["kernel_output_data_type"].append(kernel.output_dtype)
            raw_data["kernel_launching_cost"].append(kernel.host_launching_cost)
            raw_data["kernel_running_device"].append(kernel.device)
            raw_data["kernel_stream"].append(kernel.stream)
            raw_data["kernel_gap"].append(kernel.gap)
            raw_data["cpu_op_name"].append(kernel.cpu_op_name)


        df = pd.DataFrame(raw_data)
        if file_name is None:
            file_name = "test.xlsx"
        
        # 如果文件已存在，追加新 sheet；否则创建新文件
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df.to_excel(file_name, sheet_name=sheet_name, index=False)


class RocprofParser():
    def __init__(self, pftrace_file) -> None:
        self.pftrace_file = pftrace_file
        self.tp = TraceProcessor(trace=self.pftrace_file)

    def parse_kernel_dispatch(self, file_name=None):
        """
        优化版本：使用单次 SQL 查询 + PIVOT，避免循环查询
        速度提升约 100-1000 倍
        """
        # 单次 SQL 查询，使用 CASE WHEN 实现 PIVOT，一次性获取所有数据
        pivot_query = """
        SELECT 
            slice.id,
            slice.name AS kernel_name,
            MAX(CASE WHEN args.key = 'debug.delta_ns' THEN args.display_value END) AS kernel_duration,
            MAX(CASE WHEN args.key = 'debug.begin_ns' THEN args.display_value END) AS kernel_start_timestamp,
            MAX(CASE WHEN args.key = 'debug.end_ns' THEN args.display_value END) AS kernel_end_timestamp,
            MAX(CASE WHEN args.key = 'debug.agent' THEN args.display_value END) AS device_id,
            MAX(CASE WHEN args.key = 'debug.stream_ID' THEN args.display_value END) AS stream_id,
            MAX(CASE WHEN args.key = 'debug.Scratch_Size' THEN args.display_value END) AS scratch_size,
            MAX(CASE WHEN args.key = 'debug.LDS_Block_Size' THEN args.display_value END) AS lds_block_size,
            MAX(CASE WHEN args.key = 'debug.VGPR_Count' THEN args.display_value END) AS vgpr_count,
            MAX(CASE WHEN args.key = 'debug.Accum_VGPR_Count' THEN args.display_value END) AS accum_vgpr_count,
            MAX(CASE WHEN args.key = 'debug.SGPR_Count' THEN args.display_value END) AS sgpr_count,
            MAX(CASE WHEN args.key = 'debug.workgroup_size' THEN args.display_value END) AS workgroup_size,
            MAX(CASE WHEN args.key = 'debug.grid_size' THEN args.display_value END) AS grid_size
        FROM slice
        JOIN args ON slice.arg_set_id = args.arg_set_id
        WHERE slice.category = 'kernel_dispatch'
        GROUP BY slice.id
        ORDER BY slice.id
        """
        
        print("正在执行 SQL 查询（单次查询获取所有数据）...")
        result = self.tp.query(pivot_query)
        df = result.as_pandas_dataframe()
        print(f"查询完成，共 {len(df)} 条记录")
        
        # 添加 kernel_gap 列（计算同一个 stream 内相邻 kernel 的间隔）
        # kernel_gap = 当前 kernel 开始时间 - 同一 stream 内前一个 kernel 结束时间
        df['kernel_start_timestamp'] = pd.to_numeric(df['kernel_start_timestamp'], errors='coerce')
        df['kernel_end_timestamp'] = pd.to_numeric(df['kernel_end_timestamp'], errors='coerce')
        
        # 按 stream_id 分组，计算同一 stream 内的 gap
        df = df.sort_values(['stream_id', 'kernel_start_timestamp']).reset_index(drop=True)
        df['kernel_gap'] = df.groupby('stream_id').apply(
            lambda g: g['kernel_start_timestamp'] - g['kernel_end_timestamp'].shift(1)
        ).reset_index(level=0, drop=True)
        df['kernel_gap'] = df['kernel_gap'].fillna(0)  # 每个 stream 的第一个 kernel gap 为 0
        
        # 重命名和重排列以匹配原函数的输出格式
        # df = df.rename(columns={'stream_id': 'stream_id', 'grid_size': 'grid_size'})
        df = df[['kernel_name', 'kernel_duration', 'kernel_start_timestamp', 
                 'kernel_end_timestamp', 'kernel_gap', 'device_id', 'stream_id',
                 'scratch_size', 'lds_block_size', 'vgpr_count', 'accum_vgpr_count',
                 'sgpr_count', 'workgroup_size', 'grid_size']]
        
        # 保存到文件
        if file_name is None:
            file_name = "test.xlsx"
        
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name="kernel_dispatch", index=False)
        else:
            df.to_excel(file_name, sheet_name="kernel_dispatch", index=False)
        
        print(f"数据已保存到 {file_name}")
        return df
