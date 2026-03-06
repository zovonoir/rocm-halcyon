import parse
from functools import reduce
from ..KernelDef import CpuOp,Kernel

def check_mm_args(kernel,input_shape,input_dtype):
    if input_shape is None or input_dtype is None:
        raise RuntimeError(f"Got empty input or data type in an aten::mm kernel --->{kernel}")

    if len(input_shape) < 2:
        raise RuntimeError(f"Got insufficient input args for an aten::mm, expect at least 2 but actually got {len(input_shape)}: {input_shape}")

    if len(input_dtype) < 2:
        raise RuntimeError(f"Got insufficient input data type number for an aten::mm, expect at least 2 but actually got {len(input_dtype)}: {input_dtype}")
    
    if len(input_dtype) != len(input_shape):
        raise RuntimeError(f"Got unmatched input args and input data types:{input_shape = },{input_dtype = }")


no_transpose = parse.compile('''C{out_format}_Ai{lhs_format}_Bl{rhs_format}_{dtype}_{}''')
lhs_transpose = parse.compile('''C{out_format}_Al{lhs_format}_Bl{rhs_format}_{dtype}_{}''')
rhs_transpose = parse.compile('''C{out_format}_Ai{lhs_format}_Bj{rhs_format}_{dtype}_{}''')
both_transpose = parse.compile('''C{out_format}_Al{lhs_format}_Bj{rhs_format}_{dtype}_{}''')
helper_kernel = parse.compile('''C{out_format}_{}''')

def get_gemm_bpe(dtypes):
    if dtypes == 'BBS':
        return [2,2,2]
    if dtypes == 'F8BS':
        return [1,1,2]
    raise RuntimeError(f'unknown gemm output data type:{dtypes}')

def get_gemm_kernel_bpe(kernel:Kernel):
    global no_transpose
    global lhs_transpose
    global rhs_transpose
    global both_transpose
    global helper_kernel
    _prod = lambda x_list:reduce(lambda x,y:x*y,x_list)

    input_shape = kernel.input_shape
    input_data_type = kernel.input_dtype
    check_mm_args(kernel,input_shape,input_data_type)
    kernel_name = kernel.name
    # 根据实测，GEMM 中不论左支需要转置还是右支需要转置,记录到的形状都是两支都不转置
    lhs_bpe,rhs_bpe,out_bpe = 0,0,0
    if (res:=no_transpose.parse(kernel_name)) is not None:
        lhs_bpe,rhs_bpe,out_bpe = get_gemm_bpe(res['dtype'])
    elif (res:=lhs_transpose.parse(kernel_name)) is not None:
        lhs_bpe,rhs_bpe,out_bpe = get_gemm_bpe(res['dtype'])
    elif (res:=rhs_transpose.parse(kernel_name)) is not None:
        lhs_bpe,rhs_bpe,out_bpe = get_gemm_bpe(res['dtype'])
    elif (res:=both_transpose.parse(kernel_name)) is not None:
        lhs_bpe,rhs_bpe,out_bpe = get_gemm_bpe(res['dtype'])
    elif (res:=helper_kernel.parse(kernel_name)) is not None:
        lhs_bpe,rhs_bpe,out_bpe = 2,2,2
    else:
        raise RuntimeError(f'Error occured while parse gemm kernel name, unrecognized:{kernel_name}')

    return lhs_bpe,rhs_bpe,out_bpe


