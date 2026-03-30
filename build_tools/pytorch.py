# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import all_files_in_dir, cuda_version, get_cuda_include_dirs, debug_build_enabled
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    return ["torch>=2.1", "einops", "onnxscript", "onnx", "packaging", "pydantic", "nvdlfw-inspect"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/PyTorch extensions."""
    return [
        "numpy",
        "torchvision",
        "transformers",
        "torchao==0.13",
        "onnxruntime",
        "onnxruntime_extensions",
    ]


def setup_pytorch_stable_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup stable ABI extension for PyTorch support.

    This extension uses only the PyTorch stable ABI (torch/csrc/stable/),
    producing a binary that is compatible across PyTorch versions.
    It does NOT use CppExtension to avoid pulling in unstable ATen headers.
    """
    import torch

    # Source files from csrc/extensions/ directory
    stable_dir = Path(csrc_source_files) / "extensions"
    sources = all_files_in_dir(stable_dir, name_extension="cpp")
    if not sources:
        return None

    # Include directories
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
            # PyTorch headers (for stable ABI only)
            Path(torch.utils.cmake_prefix_path).parent.parent / "include",
        ]
    )

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden", "-std=c++17", "-DUSE_CUDA"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Library directories and libraries
    # Find the TE common library (libtransformer_engine.so)
    te_lib_dir = Path(csrc_source_files).parent.parent.parent
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    cuda_lib_dir = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_lib_dir):
        cuda_lib_dir = os.path.join(cuda_home, "lib")
    library_dirs = [
        str(Path(torch.utils.cmake_prefix_path).parent.parent / "lib"),
        str(te_lib_dir),
        cuda_lib_dir,
    ]
    libraries = ["torch", "torch_cpu", "c10", "cudart", "transformer_engine"]

    # Set rpath so the stable extension can find libtransformer_engine.so at runtime.
    # Use $ORIGIN for co-located libraries plus the absolute path for editable installs.
    extra_link_args = [
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{te_lib_dir.resolve()}",
    ]

    return setuptools.Extension(
        name="te_stable_abi",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args=cxx_flags,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
    )
