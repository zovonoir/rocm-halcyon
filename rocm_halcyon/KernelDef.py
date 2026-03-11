from dataclasses import dataclass
from typing import Any,Optional,List,Tuple

class Kernel:
    def __init__(self,
    name: str = None,
    duration: float = None,
    start_timestamp: float | int = None,
    end_timestamp: float | int = None,
    gap: int | float = None,
    device: int | str = None,
    stream: int = None,
    correlation: int = None,
    grid: Any | tuple | List = None,
    block: Any | tuple | List = None,
    smem: Optional[int] = None,
    host_launching_cost: Optional[float] = None,
    annotation: Optional[str] = None,
    input_shape: Optional[Any] = None,
    output_shape: Optional[Any] = None,
    input_dtype: Optional[Any] = None,
    output_dtype: Optional[Any] = None,
    input_strides: Optional[Any] = None,
    output_strides: Optional[Any] = None,
    cpu_op_name:Optional[str] = None,
    torch_op_name:Optional[str] = None,
    module_name: Optional[str] = None,
    module_type: Optional[str] = None,
    module_depth: Optional[int] = None,
    parent_module: Optional[str] = None
    ):
        self.name = name
        self.duration = duration
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.gap = gap
        self.device = device
        self.stream = stream
        self.correlation = correlation
        self.grid = grid
        self.block = block
        self.smem = smem
        self.host_launching_cost = host_launching_cost
        self.annotation = annotation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.input_strides = input_strides
        self.output_strides = output_strides
        self.cpu_op_name = cpu_op_name
        self.torch_op_name = torch_op_name
        self.module_name = module_name
        self.module_type = module_type
        self.module_depth = module_depth
        self.parent_module = parent_module

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Kernel({attrs})"

    def accept(self,visitor):
        if hasattr(visitor,"visit"):
            visitor.visit(self)
        else:
            assert 0


@dataclass
class CpuOp:
    def __init__(self,name:str,
        input_shape:Tuple = None,
        output_shape:Tuple = None,
        input_strides:Tuple = None,
        output_strides:Tuple = None,
        input_dtype:Tuple = None,
        output_dtype:Tuple = None,
        start_timestamp:float = None,
        end_timestamp:float = None,
        duration:float=None,
    ):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_strides = input_strides
        self.output_strides = output_strides
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.duration = duration

    def accept(self,visitor):
        if hasattr(visitor,"visit"):
            visitor.visit(self)
        else:
            assert 0