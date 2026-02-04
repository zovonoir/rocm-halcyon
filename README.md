This is a simple toolkit to parse torch profiler file and produce data spreadsheet.

# usage
```
import rocm_halcyon as rh

# for AMD devices
ops = rh.parse_torch_profiler("/home/zov/rocm-halcyon/torchprofiler/compare/disable/debug.json")
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
        "host_launching_cost":rh.KernelHostLaunchingCostVisitor()
    },
    file_name="debug.xlsx",sheet_name="data")


```

```
# for NV devices
ops = rh.parse_torch_profiler("/home/zov/rocm-halcyon/torchprofiler/B300/U4/wan_traces_rank0.json",device_type="nv")
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
        "host_launching_cost":rh.KernelHostLaunchingCostVisitor()
    },
    file_name="debug.xlsx",sheet_name="nv-data")
```

# known issues
Torch profiler is not always reliable, it has some chance to generate wrong duration,wrong correlation id,wrong stream relationship... It a known bug of pytorch.