from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase
from functools import reduce
from warnings import warn
from .utils import *
import parse

class VALUFloatsVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()

    def visit(self,kernel:Kernel):
        cpuop = kernel.cpu_op_name
        if cpuop == "aten::mm":
            return self.visit_aten_mm(kernel)
        return self.visit_general(kernel)
    
    def visit_aten_mm(self,kernel:Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
        check_mm_args(kernel,input_shape,input_data_type)
        lhs,rhs = input_shape[0],input_shape[1]
        M,K,N = lhs[-2],lhs[-1],rhs[-1]
        valu_floats = M*N
        return valu_floats

    def visit_general(self,kernel:Kernel):
        return 0

