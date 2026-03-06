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

        # cpuop = kernel.cpu_op_name
        # if cpuop == "aten::mm":
        #     return self.visit_aten_mm(kernel)
        # return self.visit_general(kernel)

    # def visit_aten_mm(self,kernel:Kernel):
    #     total_data_size = self.input_data_size_calculator.visit(kernel) + \
    #             self.output_data_size_calculator.visit(kernel)
    #     sec = kernel.duration / 1e6 # s
    #     BW = total_data_size / sec
    #     # print('---->',M,K,N,lhs_bpe,rhs_bpe,out_bpe,kernel.name,kernel.duration,BW,'<----')
    #     return BW

    # def visit_aten_mm(self,kernel:Kernel):
    #     input_shape = kernel.input_shape
    #     input_data_type = kernel.input_dtype
    #     check_mm_args(kernel,input_shape,input_data_type)
    #     lhs,rhs = input_shape[0],input_shape[1]
    #     kernel_name = kernel.name
    #     bs = 1 if len(lhs) == 2 else self.prod(lhs) / (lhs[-1] * lhs[-2])
    #     # 根据实测，GEMM 中不论左支需要转置还是右支需要转置,记录到的形状都是两支都不转置
    #     M,K,N = lhs[-2],lhs[-1],rhs[-1]
    #     lhs_bpe,rhs_bpe,out_bpe = 0,0,0
    #     if (res:=self.no_transpose.parse(kernel_name)) is not None:
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.lhs_transpose.parse(kernel_name)) is not None:
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.rhs_transpose.parse(kernel_name)) is not None:
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.both_transpose.parse(kernel_name)) is not None:
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.helper_kernel.parse(kernel_name)) is not None:
    #         lhs_bpe,rhs_bpe,out_bpe = 2,2,2
    #     else:
    #         raise RuntimeError(f'Error occured while parse gemm kernel name, unrecognized:{kernel_name}')
    #     assert M != 0 and N != 0 and K != 0 , f"gemm shape error:got {M=},{K=},{N=},{lhs=},{rhs=}"

    #     datasize = (lhs_bpe * M*K + rhs_bpe *K*N + out_bpe *M*N) / 1e9 # GB
    #     sec = kernel.duration / 1e6 # s
    #     BW = datasize / sec
    #     # print('---->',M,K,N,lhs_bpe,rhs_bpe,out_bpe,kernel.name,kernel.duration,BW,'<----')
    #     return BW

    # def visit_aten_mm(self,kernel:Kernel):
    #     input_shape = kernel.input_shape
    #     input_data_type = kernel.input_dtype
    #     check_mm_args(kernel,input_shape,input_data_type)
    #     lhs,rhs = input_shape[0],input_shape[1]
    #     lhs_dtype,rhs_dtype = input_data_type[0],input_data_type[1]
    #     kernel_name = kernel.name
    #     # bias = None if len(input_shape) == 2 else input_shape[2]
    #     # bias_dtype = None if len(input_data_type) == 2 else input_data_type[2]
    #     # in_bpes = self.get_bpe([lhs_dtype,rhs_dtype,bias_dtype])
    #     # out_bpe = self.get_bpe()
    #     bs = 1 if len(lhs) == 2 else self.prod(lhs) / (lhs[-1] * lhs[-2])
    #     M,K,N = 0,0,0
    #     lhs_bpe,rhs_bpe,out_bpe = 0,0,0
    #     if (res:=self.no_transpose.parse(kernel_name)) is not None:
    #         M,K,N = lhs[-2],lhs[-1],rhs[-1]
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.lhs_transpose.parse(kernel_name)) is not None:
    #         M,K,N = lhs[-1],lhs[-2],rhs[-1]
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #         print(M,K,N,lhs,rhs)
    #     elif (res:=self.rhs_transpose.parse(kernel_name)) is not None:
    #         M,K,N = lhs[-2],lhs[-1],rhs[-2]
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.both_transpose.parse(kernel_name)) is not None:
    #         M,K,N = lhs[-1],lhs[-2],rhs[-2]
    #         lhs_bpe,rhs_bpe,out_bpe = self.get_gemm_bpe(res['dtype'])
    #     elif (res:=self.helper_kernel.parse(kernel_name)) is not None:
    #         M,K,N = lhs[-2],lhs[-1],rhs[-1] # 这种情况暂时当作没有转置处理
    #         lhs_bpe,rhs_bpe,out_bpe = 2,2,2
    #     else:
    #         raise RuntimeError(f'Error occured while parse gemm kernel name, unrecognized:{kernel_name}')
    #     assert M != 0 and N != 0 and K != 0 , f"gemm shape error:got {M=},{K=},{N=},{lhs=},{rhs=}"

    #     datasize = (lhs_bpe * M*K + rhs_bpe *K*N + out_bpe *M*N) / 1e9 # GB
    #     sec = kernel.duration / 1e6 # s
    #     BW = datasize / sec
    #     return BW

    # def get_gemm_bpe(self,dtypes):
    #     if dtypes == 'BBS':
    #         return [2,2,2]
    #     if dtypes == 'F8BS':
    #         return [1,1,2]
    #     raise RuntimeError(f'unknown gemm output data type:{dtypes}')

    # def visit_general(self, kernel: Kernel):
    #     input_shape = kernel.input_shape
    #     input_data_type = kernel.input_dtype
    #     if input_shape is None or input_data_type is None:
    #         return 0
    #     bpe = self.get_bpe(input_data_type)
    #     input_size = 0
    #     for shape,size in zip(input_shape,bpe):
    #         tensor_size = 1
    #         for value in shape:
    #             if isinstance(value,int) or isinstance(value,float):
    #                 tensor_size *= value
    #             else:
    #                 break
    #         input_size += tensor_size * size
    #     output_size = input_size
    #     total_size = (output_size + input_size) / 1000000000 # byte -> GiB
    #     duration = kernel.duration / 1000 / 1000 # us -> s
    #     bw = total_size / duration
    #     return bw

    # def visit_dummy(self,kernel:Kernel):
    #     warn(f"{kernel.cpu_op_name,kernel.name} is not support yet in bw visitor!")
    #     return self.visit_general(kernel)
