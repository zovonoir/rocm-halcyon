from functools import reduce
from typing_extensions import runtime
from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase

from warnings import warn
import parse
import numpy as np
from .utils import *


"""
用于计算某个算子的输入数据字节数
"""
class InputDataSizeVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()

    def visit(self,kernel:Kernel):
        if kernel.device_type != "amd":
            return 0 # currently not support for NV input data size visitor
        cpuop = kernel.cpu_op_name
        if cpuop == "aten::mm" or cpuop == "aiter::hipb_mm":
            return self.visit_aten_mm(kernel)
        return self.visit_general(kernel)

    def visit_aten_mm(self,kernel:Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
        check_mm_args(kernel,input_shape,input_data_type)
        lhs,rhs = input_shape[0],input_shape[1]
        kernel_name = kernel.name
        bs = 1 if len(lhs) == 2 else self.prod(lhs) / (lhs[-1] * lhs[-2])
        # 根据实测，GEMM 中不论左支需要转置还是右支需要转置,记录到的形状都是两支都不转置
        M,K,N = lhs[-2],lhs[-1],rhs[-1]
        lhs_bpe,rhs_bpe,out_bpe = get_gemm_kernel_bpe(kernel)
        assert M != 0 and N != 0 and K != 0 , f"gemm shape error:got {M=},{K=},{N=},{lhs=},{rhs=}"

        input_datasize = (lhs_bpe * M*K + rhs_bpe *K*N ) # Bytes
        return input_datasize

    def visit_general(self, kernel: Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
        if input_shape is None or input_data_type is None:
            return 0
        bpe = self.get_bpe(input_data_type)
        input_size = 0
        for shape,size in zip(input_shape,bpe):
            tensor_size = 1
            for value in shape:
                if isinstance(value,int) or isinstance(value,float):
                    tensor_size *= value
                else:
                    break
            input_size += tensor_size * size
        return input_size # bytes
