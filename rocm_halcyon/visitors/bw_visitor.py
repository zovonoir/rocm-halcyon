from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase

from warnings import warn

class BWVsitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()
    
    def visit_general(self, kernel: Kernel):
        input_shape = kernel.input_shape
        input_data_type = kernel.input_dtype
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
        output_size = input_size
        total_size = (output_size + input_size) / 1000000000 # byte -> GiB
        duration = kernel.duration / 1000 / 1000 # us -> s
        bw = total_size / duration
        return bw
    
    def visit_dummy(self,kernel:Kernel):
        warn(f"{kernel.cpu_op_name,kernel.name} is not support yet in bw visitor!")
        return self.visit_general(kernel)

    
    def visit_aten_abs(self,kernel:Kernel):
        return self.visit_general(kernel)
    

    
