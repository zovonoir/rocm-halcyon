This is a simple toolkit to parse torch profiler file and produce data spreadsheet.

# usage
just modify interfaces.py
```
from AMDParser import *
from NvidiaParser import *

if __name__  == "__main__":
    # make sure your are using the right Parser
    NvidiaTorchProfilerParser("wan_traces_rank0.json").parse().export_to_excel("shape.xlsx",sheet_name="U8")
    AMDTorchProfilerParser("wan_traces_rank0.json").parse().export_to_excel("shape.xlsx",sheet_name="U4")
```

# known issues
Torch profiler is not always reliable, it has some chance to generate wrong duration,wrong correlation id,wrong stream relationship... It a known bug of pytorch.