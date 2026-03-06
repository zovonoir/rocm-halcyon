from functools import reduce
from typing_extensions import runtime
from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase

from warnings import warn
import parse
import numpy as np
from .utils import *
from .input_data_size_visitor import InputDataSizeVisitor
from .output_data_size_visitor import OutputDataSizeVisitor

"""
用于计算一个算子的平均带宽,大多数情况下,BW = (input size + output size) / duration
少数算子没有用到全部输入数据,需要专门支持
"""
class BWVsitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
        self.input_data_size_calculator = InputDataSizeVisitor()
        self.output_data_size_calculator = OutputDataSizeVisitor()


    def visit(self,kernel:Kernel):
        total_data_size = self.input_data_size_calculator.visit(kernel) + \
                self.output_data_size_calculator.visit(kernel)
        total_data_size /= 1e9 # bytes->GiB
        sec = kernel.duration / 1e6 # s
        BW = total_data_size / sec
        return BW