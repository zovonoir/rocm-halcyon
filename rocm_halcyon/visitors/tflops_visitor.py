from ..KernelDef import CpuOp,Kernel
from visitor import VisitorBase

from warnings import warn


class TflopsVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
    
    def visit_elementwise(self,kernel:Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
        input_size = 0
        for shape in (input_shape):
            tensor_size = 1
            for value in shape:
                if isinstance(value,int) or isinstance(value,float):
                    tensor_size *= value
                else:
                    break
            input_size += tensor_size
        total_size = (input_size) / 1000000000 # byte -> GiB
        duration = kernel.duration / 1000 / 1000 # us -> s
        tflops = total_size / duration
        return tflops
    
    def visit_null_op(self,*args,**kwargs):
        return 0

    def visit_dummy(self,kernel:Kernel):
        warn(f"{kernel.name},{kernel.cpu_op_name} is not supported yet!")
        return self.visit_elementwise(kernel)
    
    def _check_mnk(self,shape):
        # 这个函数无法保证顺序
        if len(shape) == 2:
            lhs,rhs = shape
        if len(shape) == 3:
            lhs,rhs,bias = shape
        else:
            raise RuntimeError(f"shape is invalid!{shape}")
        
        if len(lhs) == 3:
            b = lhs[0]
            mk = [lhs[1],lhs[2]]
            kn = [rhs[-1],rhs[-2]]
        else:
            b=1
            mk = [lhs[0],lhs[1]]
            kn = [rhs[-1],rhs[-2]]
        
        mnk = set(mk) | set(kn)
        if len(mnk) == 3:
            mnk = list(mnk)
            return b,*mnk
        elif len(mnk) == 2:
            # 这表明mk，kn中m==n
            # 这种情况下只能假设左支不用转置
            m = mk[0]
            k = mk[1]
            n = int(kn[0] * kn[1] / k)
            return b,m,k,n


    def visit_aten_mm(self,kernel:Kernel):
        b,m,k,n = self._check_mnk(kernel.input_shape)
        tfloats = b*m*n*k*2 / 1e12
        dur = kernel.duration / 1e6
        tflops = tfloats / dur

        return tflops

    def visit_aten_addmm(self,kernel:Kernel):
        return self.visit_aten_mm(kernel)

    def visit_aten_bmm(self,kernel:Kernel):
        return self.visit_aten_mm(kernel)

