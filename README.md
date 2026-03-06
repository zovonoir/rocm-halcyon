This is a simple toolkit to parse torch profiler file and produce data spreadsheet.

# install
```
 python setup.py bdist_wheel
 pip install dist/*.whl --force-reinstall
```

# usage
```
import rocm_halcyon as rh

# for AMD devices
ops = rh.parse_torch_profiler("debug.json")
rh.export_to_excel(ops,
    {
        "gpu_kernel_name":rh.GPUKernelNameVisitor(),
        "cpu_kernel_name":rh.CPUKernelNameVisitor(),
        "torch_dispatch_name":rh.TorchKernelNameVisitor(),
        "kernel_duration":rh.KernelDurationVisitor(),
        "stream":rh.KernelStreamIdVisitor(),
        "start_timestamp":rh.KernelStartTimestampVisitor(),
        "end_timestamp":rh.KernelEndTimestampVisitor(),
        "grid":rh.KernelGridSizeVisitor(),
        "block":rh.KernelBlockSizeVisitor(),
        "smem":rh.KernelSharedMemorySizeVisitor(),
        "input_shape":rh.KernelInputShapeVisitor(),
        "input_data_type":rh.KernelInputDtypeVisitor(),
        "output_shape":rh.KernelOutputShapeVisitor(),
        "output_data_type":rh.KernelOutputDtypeVisitor(),
        "device_id":rh.KernelDeviceVisitor(),
        "host_launching_cost":rh.KernelHostLaunchingCostVisitor(),
        "input_data_size":rh.InputDataSizeVisitor(),
        "output_data_size":rh.OutputDataSizeVisitor(),
        "BW(GB/s)":rh.BWVsitor(),
        "TensorCoreFloats":rh.TensorCoreFloatsVisitor(),
        "TensorCoreTFlops":rh.TensorCoreTFlopsVisitor(),
        "VALUFloats":rh.VALUFloatsVisitor(),
        "VALUTFlops":rh.VALUTFloatsVisitor(),
    },
    file_name="debug.xlsx",sheet_name="data")


```

```
# for NV devices
ops = rh.parse_torch_profiler("wan_traces_rank0.json",device_type="nv")
rh.export_to_excel(ops,
    {
        "gpu_kernel_name":rh.GPUKernelNameVisitor(),
        "cpu_kernel_name":rh.CPUKernelNameVisitor(),
        "torch_dispatch_name":rh.TorchKernelNameVisitor(),
        "kernel_duration":rh.KernelDurationVisitor(),
        "stream":rh.KernelStreamIdVisitor(),
        "start_timestamp":rh.KernelStartTimestampVisitor(),
        "end_timestamp":rh.KernelEndTimestampVisitor(),
        "grid":rh.KernelGridSizeVisitor(),
        "block":rh.KernelBlockSizeVisitor(),
        "smem":rh.KernelSharedMemorySizeVisitor(),
        "input_shape":rh.KernelInputShapeVisitor(),
        "input_data_type":rh.KernelInputDtypeVisitor(),
        "output_shape":rh.KernelOutputShapeVisitor(),
        "output_data_type":rh.KernelOutputDtypeVisitor(),
        "device_id":rh.KernelDeviceVisitor(),
        "host_launching_cost":rh.KernelHostLaunchingCostVisitor(),
        "input_data_size":rh.InputDataSizeVisitor(),
        "output_data_size":rh.OutputDataSizeVisitor(),
        "BW(GB/s)":rh.BWVsitor(),
        "TensorCoreFloats":rh.TensorCoreFloatsVisitor(),
        "TensorCoreTFlops":rh.TensorCoreTFlopsVisitor(),
        "VALUFloats":rh.VALUFloatsVisitor(),
        "VALUTFlops":rh.VALUTFloatsVisitor(),
    },
    file_name="debug.xlsx",sheet_name="nv-data")
```

# known issues
Torch profiler is not always reliable, it has some chance to generate wrong duration,wrong correlation id,wrong stream relationship... It a known bug of pytorch.
Currently, Operator level analysis is not fully support, only support gemm, and some elementwise kernel, use with cautious.