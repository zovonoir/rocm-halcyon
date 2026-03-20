from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase
from functools import reduce
from warnings import warn
from .utils import *

"""
这个类用于计算所有需要使用到tensor core的算子的Tensor Core理论浮点运算量
"""
class TensorCoreFloatsVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
        self.prod = lambda x_list:reduce(lambda x,y:x*y,x_list)
    
    def visit(self,kernel:Kernel):
        if kernel.device_type != "amd":
            return 0 # currently not support for NV input data size visitor
        cpuop = kernel.cpu_op_name
        if cpuop == "aten::mm":
            return self.visit_aten_mm(kernel)
        return self.visit_general(kernel)

    def visit_aten_mm(self,kernel:Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
        check_mm_args(kernel,input_shape,input_data_type)
        lhs,rhs = input_shape[0],input_shape[1]
        # 根据实测，GEMM 中不论左支需要转置还是右支需要转置,记录到的形状都是两支都不转置
        M,K,N = lhs[-2],lhs[-1],rhs[-1]
        bs = 1 if len(lhs) == 2 else self.prod(lhs) / (lhs[-1] * lhs[-2])
        tfloats = 2*M*K*N*bs
        return tfloats

    def visit_general(self,kernel:Kernel):
        return 0

