"""
Example: Profiling PyTorch models with module architecture information.

This example demonstrates how to use ModuleAnnotator from rocm-halcyon to profile a PyTorch model and export kernel metrics with
module-level context (which module/layer generated each kernel).

This enables users to:
1. Understand which model components are performance bottlenecks
2. Aggregate metrics by module type or hierarchy level
3. Correlate low-level kernel operations with high-level model architecture

Requirements:
    pip install torch rocm-halcyon openpyxl
"""

import torch
import torch.nn as nn

import rocm_halcyon as rh
from rocm_halcyon import ModuleAnnotator

# A simplified Transformer Encoder Layer to demonstrate profiling
class SimpleTransformerLayer(nn.Module):
    """
    A basic implementation of a Transformer Encoder Layer.
    It includes Multi-Head Self-Attention, Layer Normalization, and a Feed-Forward Network (MLP).
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-Forward Network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass for the Transformer Layer.
        We will wrap key sections with record_function to create distinct profiler labels.
        """
        # 1. Self-Attention Block
        with torch.profiler.record_function("Self-Attention"):
            # The multi-head attention layer expects query, key, and value inputs.
            # For self-attention, the source tensor is used for all three.
            attn_output, _ = self.self_attention(src, src, src)

        # Residual connection with dropout and LayerNorm
        with torch.profiler.record_function("Add & Norm 1"):
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

        # 2. MLP (Feed-Forward) Block
        with torch.profiler.record_function("MLP"):
            mlp_output = self.mlp(src)

        # Residual connection with dropout and LayerNorm
        with torch.profiler.record_function("Add & Norm 2"):
            src = src + self.dropout2(mlp_output)
            src = self.norm2(src)

        return src



def profile_with_module_info():
    """Profile a model with module annotations and export to Excel."""

    # Create model and sample inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model and Input Configuration ---
    batch_size = 16
    seq_length = 128
    embed_dim = 512  # d_model
    num_heads = 8
    ff_dim = 2048   # Hidden dimension in MLP

    inputs = torch.randn(batch_size, seq_length, embed_dim).to(device)
    model = SimpleTransformerLayer(embed_dim, num_heads, ff_dim).to(device)
    model.eval() # Set to evaluation mode for profiling inference

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # Profile with module annotations
    print("Profiling with module annotations...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        # Use ModuleAnnotator to inject module information
        with ModuleAnnotator(model):
            output = model(inputs)

    # Export trace file
    trace_file = "trace_with_modules.json"
    prof.export_chrome_trace(trace_file)
    print(f"Trace exported to: {trace_file}")

    # Parse trace with rocm-halcyon
    print("Parsing trace file...")
    device_type = "nv" if torch.cuda.is_available() else "amd"
    kernels = rh.parse_torch_profiler(trace_file, device_type=device_type)

    # Export to Excel with module information columns
    print("Exporting to Excel...")
    excel_file = "analysis_with_modules.xlsx"

    rh.export_to_excel(kernels, {
        # GPU/CPU operation names
        "gpu_kernel_name": rh.GPUKernelNameVisitor(),
        "cpu_op_name": rh.CPUKernelNameVisitor(),

        # NEW: Module architecture information
        "module_name": rh.ModuleNameVisitor(),           # Full module path
        "module_type": rh.ModuleTypeVisitor(),           # Module class name
        "module_hierarchy": rh.ModuleHierarchyVisitor(), # Readable hierarchy
        "module_depth": rh.ModuleDepthVisitor(),         # Nesting level
        "parent_module": rh.ParentModuleVisitor(),       # Parent module path

        # Performance metrics
        "duration_us": rh.KernelDurationVisitor(),
        "bandwidth_GB/s": rh.BWVsitor(),

        # Kernel configuration
        "grid": rh.KernelGridSizeVisitor(),
        "block": rh.KernelBlockSizeVisitor(),
        "smem": rh.KernelSharedMemorySizeVisitor(),

        # Data shapes and types
        "input_shape": rh.KernelInputShapeVisitor(),
        "input_dtype": rh.KernelInputDtypeVisitor(),
        "output_shape": rh.KernelOutputShapeVisitor(),
        "output_dtype": rh.KernelOutputDtypeVisitor(),

        # Timing information
        "start_timestamp": rh.KernelStartTimestampVisitor(),
        "end_timestamp": rh.KernelEndTimestampVisitor(),
        "gap_us": rh.KernelGapVisitor(),

        # Device information
        "stream": rh.KernelStreamIdVisitor(),
        "device": rh.KernelDeviceVisitor(),
    }, file_name=excel_file, sheet_name="kernels_with_modules")

    print(f"Analysis exported to: {excel_file}")
    print("\nThe Excel file now contains module information columns:")
    print("  - module_name: Full module path (e.g., 'model.encoder.0')")
    print("  - module_type: Module class (e.g., 'Linear', 'ReLU')")
    print("  - module_hierarchy: Readable format (e.g., 'model → encoder → 0')")
    print("  - module_depth: Nesting level in model hierarchy")
    print("  - parent_module: Parent module path")
    print("\nYou can now:")
    print("  1. Filter kernels by module type (e.g., all Linear layers)")
    print("  2. Aggregate metrics by module (e.g., total time per module)")
    print("  3. Identify performance bottlenecks at module level")


def profile_without_module_info():
    """
    Profile without module annotations (backward compatible).

    This shows that the new functionality is optional - existing code
    continues to work, and module fields will be empty/None.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    inputs = torch.randn(32, 128).to(device)

    print("\nProfiling WITHOUT module annotations (backward compatible)...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
    ) as prof:
        # Notice: No ModuleAnnotator here
        output = model(inputs)

    trace_file = "trace_without_modules.json"
    prof.export_chrome_trace(trace_file)

    device_type = "nv" if torch.cuda.is_available() else "amd"
    kernels = rh.parse_torch_profiler(trace_file, device_type=device_type)

    excel_file = "analysis_without_modules.xlsx"
    rh.export_to_excel(kernels, {
        "gpu_kernel_name": rh.GPUKernelNameVisitor(),
        "cpu_op_name": rh.CPUKernelNameVisitor(),
        "module_name": rh.ModuleNameVisitor(),  # Will be None/empty
        "module_type": rh.ModuleTypeVisitor(),  # Will be None/empty
        "duration_us": rh.KernelDurationVisitor(),
    }, file_name=excel_file, sheet_name="kernels")

    print(f"Analysis exported to: {excel_file}")
    print("Module fields will be empty (backward compatible)")


if __name__ == "__main__":
    # Main example: with module information
    profile_with_module_info()

    # Optional: demonstrate backward compatibility
    # profile_without_module_info()
