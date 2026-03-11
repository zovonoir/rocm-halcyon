"""
Profiler bridge for annotating PyTorch modules during profiling.

This module provides ModuleAnnotator, a context manager that wraps nn.Module.forward()
methods to inject torch.profiler.record_function() annotations with module metadata.
These annotations can then be extracted by profiling tools like rocm-halcyon to provide
model architecture context in performance analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import functools


class ModuleAnnotator:
    """
    Context manager that annotates PyTorch modules during profiling.

    Wraps each nn.Module.forward() method to add profiler annotations with module
    metadata (name, type, depth, parent). This allows profiling tools to correlate
    low-level kernel operations with high-level model architecture.

    Usage:
        model = MyModel()
        inputs = torch.randn(...)

        with torch.profiler.profile(...) as prof:
            with ModuleAnnotator(model):
                output = model(inputs)

        prof.export_chrome_trace("trace.json")

    The trace will contain user_annotation events with format:
        "MODULE:{module_path}|{module_type}|depth={depth}|parent={parent}"

    Args:
        model: The PyTorch model to annotate
        trace_depth: Maximum depth to trace (None = trace all depths)
    """

    def __init__(self, model: nn.Module, trace_depth: Optional[int] = None):
        self.model = model
        self.trace_depth = trace_depth
        self._original_forwards: Dict[nn.Module, Callable] = {}
        self._module_metadata: Dict[nn.Module, Dict[str, Any]] = {}

    def __enter__(self):
        """Wrap all module forward methods with profiler annotations."""
        self._build_module_metadata()
        self._wrap_modules()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original forward methods."""
        self._unwrap_modules()
        return False

    def _build_module_metadata(self):
        """Build metadata for each module (name, type, depth, parent)."""
        def traverse(module: nn.Module, name: str, depth: int, parent: str):
            # Store metadata for this module
            self._module_metadata[module] = {
                'name': name,
                'type': type(module).__name__,
                'depth': depth,
                'parent': parent
            }

            # Recursively traverse child modules
            for child_name, child_module in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                traverse(child_module, child_full_name, depth + 1, name)

        # Start traversal from root with empty name
        traverse(self.model, 'model', 0, '')

    def _wrap_modules(self):
        """Wrap forward methods of all modules with profiler annotations."""
        for module, metadata in self._module_metadata.items():
            # Skip if we've reached the trace depth limit
            if self.trace_depth is not None and metadata['depth'] > self.trace_depth:
                continue

            # Store original forward method
            original_forward = module.forward
            self._original_forwards[module] = original_forward

            # Create wrapped forward with annotation
            def make_wrapped_forward(mod, meta, orig_forward):
                @functools.wraps(orig_forward)
                def wrapped_forward(*args, **kwargs):
                    # Build annotation string
                    annotation = (
                        f"MODULE:{meta['name']}|{meta['type']}|"
                        f"depth={meta['depth']}|parent={meta['parent']}"
                    )

                    # Execute forward within profiler annotation
                    with torch.profiler.record_function(annotation):
                        return orig_forward(*args, **kwargs)

                return wrapped_forward

            # Replace forward method with wrapped version
            module.forward = make_wrapped_forward(module, metadata, original_forward)

    def _unwrap_modules(self):
        """Restore original forward methods."""
        for module, original_forward in self._original_forwards.items():
            module.forward = original_forward

        # Clear stored data
        self._original_forwards.clear()
        self._module_metadata.clear()
