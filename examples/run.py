import rocm_halcyon as rh

ops = rh.parse_torch_profiler("/home/zov/rocm-halcyon/torchprofiler/1773129217.1439552-TP-0.trace.json",device_type="amd")

rh.export_to_excel(ops,
    {
        "gpu_kernel_name":rh.GPUKernelNameVisitor(),
        "cpu_kernel_name":rh.CPUKernelNameVisitor(),
        "kernel_duration":rh.KernelDurationVisitor(),
        "stream":rh.KernelStreamIdVisitor(),
        "start_timestamp":rh.KernelStartTimestampVisitor(),
        "end_timestamp":rh.KernelEndTimestampVisitor(),
        "grid":rh.KernelGridSizeVisitor(),
        "block":rh.KernelBlockSizeVisitor(),
        "smem":rh.KernelSharedMemorySizeVisitor(),
        "input_shape":rh.KernelInputShapeVisitor(),
        "input_data_type":rh.KernelInputDtypeVisitor(),
        "input_data_size":rh.InputDataSizeVisitor(),
        "output_data_size":rh.OutputDataSizeVisitor(),
        "BW(GB/s)":rh.BWVsitor(),
        "TensorCoreFloats":rh.TensorCoreFloatsVisitor(),
        "TensorCoreTFlops":rh.TensorCoreTFlopsVisitor(),
        "VALUFloats":rh.VALUFloatsVisitor(),
        "VALUTFlops":rh.VALUTFloatsVisitor(),
        "output_shape":rh.KernelOutputShapeVisitor(),
        "output_data_type":rh.KernelOutputDtypeVisitor(),
        "device_id":rh.KernelDeviceVisitor(),
        "host_launching_cost":rh.KernelHostLaunchingCostVisitor()
    },
    file_name="sglang-fp8-baseline-shape.xlsx",sheet_name="your-sheet-name")