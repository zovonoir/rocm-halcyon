"""
Visitor classes for extracting module architecture information from kernels.

These visitors extract module-level information that was injected during profiling
using the ModuleAnnotator context manager from torchvista.
"""

from .visitor import VisitorBase
from ..KernelDef import Kernel
from typing import Optional


class ModuleNameVisitor(VisitorBase):
    """
    Extracts the full module path/name for a kernel.

    Returns the hierarchical module path (e.g., "model.encoder.layer.0.attention")
    that identifies which module in the model generated this kernel.
    """

    def visit(self, kernel: Kernel) -> Optional[str]:
        return kernel.module_name


class ModuleTypeVisitor(VisitorBase):
    """
    Extracts the module type/class name for a kernel.

    Returns the class name of the module (e.g., "Linear", "MultiheadAttention")
    that generated this kernel.
    """

    def visit(self, kernel: Kernel) -> Optional[str]:
        return kernel.module_type


class ModuleDepthVisitor(VisitorBase):
    """
    Extracts the module depth in the model hierarchy.

    Returns the nesting level of the module (0 for root, 1 for direct children, etc.)
    This helps understand the structural depth of the module that generated the kernel.
    """

    def visit(self, kernel: Kernel) -> Optional[int]:
        return kernel.module_depth


class ParentModuleVisitor(VisitorBase):
    """
    Extracts the parent module name for a kernel.

    Returns the full path of the parent module that contains the module
    which generated this kernel. Useful for understanding module relationships.
    """

    def visit(self, kernel: Kernel) -> Optional[str]:
        return kernel.parent_module


class ModuleHierarchyVisitor(VisitorBase):
    """
    Formats the module path as a readable hierarchy.

    Converts dot-separated module paths (e.g., "model.encoder.layer.0")
    into readable arrow-separated hierarchies (e.g., "model → encoder → layer → 0")
    for better visualization in spreadsheets.
    """

    def visit(self, kernel: Kernel) -> Optional[str]:
        if kernel.module_name:
            return " → ".join(kernel.module_name.split('.'))
        return None
