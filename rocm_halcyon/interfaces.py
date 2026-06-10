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
from .AMDParser import AMDTorchProfilerParser
from .NvidiaParser import NvidiaTorchProfilerParser
from .KernelDef import Kernel
from . import halcyon_core as _C

def say_hello():
    """调用 C++ 扩展模块中的 say_hello 函数"""
    _C.say_hello()


class VisitorBase():
    def __init__(self,formatter = None) -> None:
        self.formatter = formatter
    def visit(self,):
        assert 0,"call Virtual Base Visit is illegal!"

class GPUKernelNameVisitor(VisitorBase):
    """用于获取GPU实际执行的kernel名称"""
    def visit(self,kernel:Kernel):
        return kernel.name

class CPUKernelNameVisitor(VisitorBase):
    """用于获取某个算子的CPU侧的下发名称,如aten::mm"""
    def visit(self,kernel:Kernel):
        return kernel.cpu_op_name

class TorchKernelNameVisitor(VisitorBase):
    """用于获取某个算子的pytorch层面的执行名称,如torch.nn.Linear"""
    def visit(self,kernel:Kernel):
        return kernel.torch_op_name

class KernelDurationVisitor(VisitorBase):
    """用于获取某个算子在GPU上的执行时间,单位us"""
    def visit(self,kernel:Kernel):
        return kernel.duration

class KernelStartTimestampVisitor(VisitorBase):
    """用于获取某个算子的GPU上的开始时间戳,仅为数值,无单位"""
    def visit(self,kernel:Kernel):
        return kernel.start_timestamp

class KernelEndTimestampVisitor(VisitorBase):
    """用于获取某个算子在GPU上的结束时间戳,仅为数值,无单位"""
    def visit(self,kernel:Kernel):
        return kernel.end_timestamp

class KernelGapVisitor(VisitorBase):
    """用于获取第n个算子的开始时间与第n-1个算子的结束时间的差值,单位us"""
    def visit(self,kernel:Kernel):
        return kernel.gap

class KernelGridSizeVisitor(VisitorBase):
    """用于获取算子启动时的grid配置"""
    def visit(self,kernel:Kernel):
        return kernel.grid

class KernelBlockSizeVisitor(VisitorBase):
    """用于获取算子启动时的block配置"""
    def visit(self,kernel:Kernel):
        return kernel.block

class KernelStreamIdVisitor(VisitorBase):
    """用于获取承载该算子在GPU上运行的stream id"""
    def visit(self,kernel:Kernel):
        return kernel.stream

class KernelSharedMemorySizeVisitor(VisitorBase):
    """用于获取该算子启动时预先配置的共享内存大小"""
    def visit(self,kernel:Kernel):
        return kernel.smem

class KernelInputShapeVisitor(VisitorBase):
    """用于获取kernel在CPU侧记录的输入形状"""
    def visit(self,kernel:Kernel):
        return kernel.input_shape

class KernelOutputShapeVisitor(VisitorBase):
    """用于获取kernel在CPU侧记录到的输出形状"""
    def visit(self,kernel:Kernel):
        return kernel.output_shape

class KernelInputDtypeVisitor(VisitorBase):
    """用于获取kenrel在CPU侧记录到的输入数据类型"""
    def visit(self,kernel:Kernel):
        return kernel.input_dtype

class KernelOutputDtypeVisitor(VisitorBase):
    """用于获取kernel在CPU侧记录到的输出数据类型"""
    def visit(self,kernel:Kernel):
        return kernel.output_dtype

class KernelHostLaunchingCostVisitor(VisitorBase):
    """用于获取启动该算子的那条launch kernel API调用的耗时,单位us"""
    def visit(self,kernel:Kernel):
        return kernel.host_launching_cost

class KernelDeviceVisitor(VisitorBase):
    """用于或者承载该算子运行的GPU的ID"""
    def visit(self,kernel:Kernel):
        return kernel.device

class UserAnnotationVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.user_annotation

def parse_torch_profiler(datasource:str,device_type="amd"):
    device_type = device_type.lower()
    assert device_type in ["amd","nvidia","nv"]
    Parser = AMDTorchProfilerParser
    if device_type != "amd":
        Parser = NvidiaTorchProfilerParser
    parser = Parser(datasource)
    kernels = parser.parse()

    return kernels

def export_to_excel(kernels:List[Kernel],visitors:Dict[str,VisitorBase],file_name = None,sheet_name = None):
    data = dict()
    for header,visitor in visitors.items():
        header_data = []
        for kernel in tqdm.tqdm(kernels,desc=f"processing:{header}"):
            result = visitor.visit(kernel)
            header_data.append(result)
        data[header] = header_data
    df = pd.DataFrame(data)
    if file_name is None:
        file_name = "test.xlsx"
    if sheet_name == None:
        sheet_name = "data"

    def make_sheet_name(base_name, suffix):
        suffix = f"_{suffix}"
        return f"{base_name[:31 - len(suffix)]}{suffix}"

    def write_dataframe(writer):
        try:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        except ValueError as error:
            if "This sheet is too large" not in str(error):
                raise

            max_rows_per_sheet = 1048575
            for i, start in enumerate(range(0, len(df), max_rows_per_sheet), start=1):
                chunk = df.iloc[start:start + max_rows_per_sheet]
                chunk.to_excel(writer, sheet_name=make_sheet_name(sheet_name, i), index=False)

    if os.path.exists(file_name):
        with pd.ExcelWriter(file_name, 
                            mode='a', 
                            engine='openpyxl', 
                            if_sheet_exists='replace') as writer:
            write_dataframe(writer)
    else:
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            write_dataframe(writer)
    