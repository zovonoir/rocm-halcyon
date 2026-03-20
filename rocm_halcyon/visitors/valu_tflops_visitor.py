from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase
from functools import reduce
from warnings import warn
from .utils import *
import parse
from .valu_floats_visitor import VALUFloatsVisitor

"""
用于计算某个算子的向量VALU单元的实际算力值,一般情况下,这等于 VALU运算量 / duration
"""
class VALUTFloatsVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
        self.valu_calculator = VALUFloatsVisitor()

    def visit(self,kernel:Kernel):
        if kernel.device_type != "amd":
            return 0 # currently not support for NV input data size visitor
        cpuop = kernel.cpu_op_name
        if cpuop == "aten::mm":
            return self.visit_aten_mm(kernel)
        return self.visit_general(kernel)
    
    def visit_aten_mm(self,kernel:Kernel):
        valu_floats = self.valu_calculator.visit(kernel) / 1e12
        sec = kernel.duration / 1e6 # us -> s

        return valu_floats / sec

    def visit_general(self,kernel:Kernel):
        return 0

