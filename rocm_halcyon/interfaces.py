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
    def visit(self,kernel:Kernel):
        return kernel.name

class CPUKernelNameVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.cpu_op_name

class TorchKernelNameVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.torch_op_name

class KernelDurationVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.duration

class KernelStartTimestampVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.start_timestamp

class KernelEndTimestampVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.end_timestamp

class KernelGapVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.gap

class KernelGridSizeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.grid

class KernelBlockSizeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.block

class KernelStreamIdVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.stream

class KernelSharedMemorySizeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.smem

class KernelInputShapeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.input_shape

class KernelOutputShapeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.output_shape

class KernelInputDtypeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.input_dtype

class KernelOutputDtypeVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.output_dtype

class KernelHostLaunchingCostVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.host_launching_cost

class KernelDeviceVisitor(VisitorBase):
    def visit(self,kernel:Kernel):
        return kernel.device


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
        for kernel in kernels:
            result = visitor.visit(kernel)
            header_data.append(result)
        data[header] = header_data
    df = pd.DataFrame(data)
    if file_name is None:
        file_name = "test.xlsx"
    if sheet_name == None:
        sheet_name = "data"

    if os.path.exists(file_name):
        with pd.ExcelWriter(file_name, 
                            mode='a', 
                            engine='openpyxl', 
                            if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df.to_excel(file_name, sheet_name=sheet_name, index=False)
    