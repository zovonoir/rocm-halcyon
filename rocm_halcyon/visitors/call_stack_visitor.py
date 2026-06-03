from functools import reduce
from typing_extensions import runtime
from ..KernelDef import CpuOp,Kernel
from .visitor import VisitorBase

from warnings import warn
import re
import parse
import numpy as np
from .utils import *


"""
用于获取某个算子的调用栈
"""
class CallStackVisitor(VisitorBase):
    def __init__(self) -> None:
        super().__init__()

    def visit(self,kernel:Kernel):
        call_stack = kernel.call_stack
        if not call_stack:
            return ''

        call_stack = [item for item in call_stack
                      if not re.search(r'\(\d+\)', item) and not re.match(r'^<.*>$', item)]
        if not call_stack:
            return ''
        
        call_stack = list(map(lambda s:s.replace('nn.Module: ',''),call_stack))
        if not call_stack:
            return ''
        
        return reduce(lambda x,y:x+'➡'+y,call_stack)
