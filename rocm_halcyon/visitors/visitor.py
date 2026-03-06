from ..KernelDef import Kernel
class VisitorBase():
    def __init__(self) -> None:
        self.dtypes_to_bpe = {
            'long int':8,
            'bool':1,
            'int':4,
            'Scalar':0,
            'c10::BFloat16':2,
            'float':4,
            '':0,
            'TensorList':0,
            'ScalarList':0,
            'double':8,
            None:0
        }
    def get_bpe(self,dtypes):
        if isinstance(dtypes,list):
            res = []
            for dtype in dtypes:
                res.append(self.dtypes_to_bpe[dtype])
            return res
        else:
            return self.dtypes_to_bpe[dtypes]


    def visit_aten_abs(self,kernel:Kernel):
        pass
