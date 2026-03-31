from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase
from functools import reduce
from warnings import warn
from .utils import *
from .tensor_core_floats_visitor import TensorCoreFloatsVisitor

"""
用于计算某个算子的TensorCore实际算力值,单位TFlops,一般情况下,这等于 TensorCore浮点运算量 / duration
"""
class TensorCoreTFlopsVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
        self.prod = lambda x_list:reduce(lambda x,y:x*y,x_list)
        self.floats_calculator = TensorCoreFloatsVisitor()

    def visit(self,kernel:Kernel):
        if kernel.device_type != "amd":
            return 0 # currently not support for NV input data size visitor
        cpuop = kernel.cpu_op_name
        if cpuop == "aten::mm" or cpuop == "aiter::hipb_mm":
            return self.visit_aten_mm(kernel)
        return self.visit_general(kernel)
    
    def visit_aten_mm(self,kernel:Kernel):
        tfloats = self.floats_calculator.visit(kernel) / 1e12 # T
        sec = kernel.duration / 1e6 # us -> s
        TFlops = tfloats / sec
        return TFlops

    def visit_general(self,kernel:Kernel):
        return 0

