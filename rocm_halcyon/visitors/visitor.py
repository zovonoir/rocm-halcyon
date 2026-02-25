from ..KernelDef import Kernel
class VisitorBase():
    def __init__(self) -> None:
        self.dtypes_to_bpe = {
            'long int':8,
            'bool':1,
            'int':4,
            'Scalar':0,
            'c10:Bfloat16':2,
            'float':4,
        }
    def get_bpe(self,dtypes):
        if isinstance(dtypes,list):
            res = []
            for dtype in dtypes:
                res.append(self.dtypes_to_bpe[dtype])
        else:
            return self.dtypes_to_bpe[dtypes]


    def visit_aten_abs(self,kernel:Kernel):
        pass
