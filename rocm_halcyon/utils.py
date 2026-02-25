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
import json


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


def load_all_events(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        all_events = data.get("traceEvents", [])
        if all_events == []:
            assert 0,"json file is empty."
    return all_events

