import json



from ast import Tuple
from fileinput import filename
import os
import pandas as pd
import numpy as np
import json
import tqdm
from warnings import warn as warning
from typing import List 
from dataclasses import dataclass
from typing import Optional, Any
import parse
import re
from perfetto.trace_processor import TraceProcessor
# @dataclass
# class Kernel:
#     name: str
#     duration: float
#     start_timestamp: float | int
#     end_timestamp: float | int
#     gap: int | float
#     device: int | str
#     stream: int
#     correlation: int
#     grid: Any | tuple | List
#     block: Any | tuple | List
#     smem: Optional[int]
#     host_launching_cost: Optional[float]
#     annotation: Optional[str]
#     input_shape: Optional[Any] = None
#     output_shape: Optional[Any] = None
#     input_dtype: Optional[Any] = None
#     output_dtype: Optional[Any] = None
#     input_strides: Optional[Any] = None
#     output_strides: Optional[Any] = None
#     cpu_op_name:str = None

#     def __str__(self):
#         debug_str = f"Kernel Object:\n"
#         debug_str += f"    Kernel name:{self.name}\n"
#         debug_str += f"    Input shape:{self.input_shape},dtype:{self.input_dtype}\n"
#         debug_str += f"    Output shape:{self.output_shape},dtype:{self.output_dtype}\n"
#         debug_str += f"    Grid:{self.grid},Block:{self.block},Shared Memory:{self.smem}\n"
#         debug_str += f"    Start timestamp:{self.start_timestamp},end timestamp:{self.end_timestamp},duration(us):{self.duration}\n"
#         debug_str += f"    Stream:{self.stream}\n"
#         return debug_str

# @dataclass
# class CpuOp:
#     name:str
#     input_shape:Tuple = None
#     output_shape:Tuple = None
#     input_strides:Tuple = None
#     output_strides:Tuple = None
#     input_dtype:Tuple = None
#     output_dtype:Tuple = None
#     start_timestamp:float = None
#     end_timestamp:float = None
#     duration:float=None




def shirink_profiler(json_file,keeps=["kernel","cpu_op"],keep_items = None):
    with open(json_file,'r') as f:
        data = json.load(f)
        data  = data.get("traceEvents",[])
        shrinked_result = []
        if data != []:
            for keep in keeps:
                keep_count = 0
                for item in data:
                    if "cat" in item and item["cat"]==keep:
                        shrinked_result.append(item)
                        keep_count += 1
                        if keep_items is not None and keep_count == keep_items:
                            break
        with open('shrinked_output.json', 'w', encoding='utf-8') as f:
            json.dump(shrinked_result, f, indent=2, ensure_ascii=False)



