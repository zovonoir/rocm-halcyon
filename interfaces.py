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
    NvidiaTorchProfilerParser("wan_traces_rank0.json").parse().export_to_excel("shape.xlsx",sheet_name="NV-U8")
    AMDTorchProfilerParser("wan_traces_rank0.json").parse().export_to_excel("shape.xlsx",sheet_name="AMD-U4")
