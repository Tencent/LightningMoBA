# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
# Copyright (c) 2025, FlashMoBA Team.

import sys
import functools
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)


long_description = (Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "lightning_moba"

# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
SKIP_CUDA_BUILD = os.getenv("LIGHTNING_MOBA_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("LIGHTNING_MOBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"

@functools.lru_cache(maxsize=None)
def cuda_archs() -> list[str]:
    return os.getenv("LIGHTNING_MOBA_CUDA_ARCHS", "80;90a").split(";")


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


ext_modules = []

# CUTLASS headers are vendored with this source tree.
if not (Path(this_dir) / "csrc/cutlass/include/cutlass/cutlass.h").is_file():
    raise FileNotFoundError("csrc/cutlass headers are missing")

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

    check_if_cuda_home_none("lightning_moba")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "LightningMoBA is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8") and "90" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if "90a" in cuda_archs():
            if bare_metal_version < Version("12.3"):
                raise RuntimeError("The default SM90 pipeline requires CUDA 12.3+ for modifiable TMA descriptors")
            # Hopper architecture-specific target required by WGMMA instructions.
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90a,code=sm_90a")
            cc_flag.append("-DLIGHTNING_MOBA_SM90_PIPELINE_AVAILABLE=1")
            cc_flag.append("-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1")
            # CUDA 12.6 does not consistently surface
            # __CUDA_ARCH_FEAT_SM90_ALL for this extension's mixed SM80/SM90a
            # build. Enable modifiable TMA descriptors for dynamic packed
            # offsets in the Hopper-specific image.
            cc_flag.append("-DCUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED=1")
        if bare_metal_version >= Version("12.8") and "100" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]

    ext_modules.append(
        CUDAExtension(
            name="lightning_moba_cuda",
            sources=[
                "csrc/lightning_moba_binding.cpp",
                "csrc/block8/block8_api.cpp",
                "csrc/block8/attention.cu",
                "csrc/block8/layout_probe.cu",
                "csrc/lightning_moba/lightning_moba_api.cpp",
                "csrc/flash_dynamic/flash_dynamic_api.cu",
                "csrc/flash_dynamic/flash_dynamic_attention_api.cu",
                # LightningMoBA
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim32_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim32_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim64_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim64_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim96_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim96_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim128_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim128_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim192_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim192_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim256_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim256_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_hdim256_bf16_causal_sm80.cu",
                # KV-Centric forward kernels
                "csrc/lightning_moba/src/flash_kvcentric_worklist.cu",
                "csrc/lightning_moba/src/flash_kvcentric_stream.cu",
                "csrc/lightning_moba/src/sm90_kv_centric_pipeline.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_kvcentric_hdim128_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_kvcentric_hdim128_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_kvcentric_hdim128_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_fwd_kvcentric_hdim128_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_combine.cu",

                "csrc/lightning_moba/src/lightning_moba_bwd_hdim32_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim32_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim64_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim64_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim96_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim96_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim128_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim128_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim192_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim192_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim256_fp16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim256_bf16_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/lightning_moba/src/lightning_moba_bwd_hdim256_bf16_causal_sm80.cu",
                
                "csrc/flash_topk/flash_topk_api.cpp",
                # VARLEN SORT
                "csrc/flash_topk/src/varlen_sort.cu",
                #MOBA TOPK
                "csrc/flash_topk/src/flash_topk16_hdim64_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim64_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim128_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim128_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim256_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim256_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim64_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim64_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim128_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim128_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim256_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim256_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim64_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim64_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim128_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim128_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim256_fp16_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim256_bf16_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk16_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk32_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_topk/src/flash_topk64_hdim256_bf16_causal_sm80.cu",
                
            ],
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "lightning_moba",
                Path(this_dir) / "csrc" / "lightning_moba" / "src",
                Path(this_dir) / "csrc" / "flash_topk",
                Path(this_dir) / "csrc" / "flash_topk" / "src",
                Path(this_dir) / "csrc" / "flash_dynamic" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "lightning_moba" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("LIGHTNING_MOBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    py_modules=["lightning_moba_dynamic_cuda"],
    packages=find_packages(include=("lightning_moba", "lightning_moba.*")),
    author="LightningMoBA contributors",
    description="LightningMoBA",
    license="BSD-3-Clause",
    license_files=("LICENSE", "csrc/cutlass/LICENSE.txt", "csrc/cutlass/EULA.txt"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
        "triton",
        "packaging",
    ],
    extras_require={"test": ["pytest>=7"]},
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
