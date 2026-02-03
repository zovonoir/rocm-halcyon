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
from AMDParser import *
from NvidiaParser import *

if __name__  == "__main__":
    NvidiaTorchProfilerParser("/home/zov/rocm-halcyon/torchprofiler/B300/U8/wan_traces_rank0.json").parse().export_to_excel("B300-shape.xlsx",sheet_name="U8")
    NvidiaTorchProfilerParser("/home/zov/rocm-halcyon/torchprofiler/B300/U4/wan_traces_rank0.json").parse().export_to_excel("B300-shape.xlsx",sheet_name="U4")
