# Compression branch build artifacts

The `Build compression artifacts` GitHub Actions workflow packages optimized
PLINK 2 and conditional-rANS tools without changing upstream's default build
configuration. It runs manually or when a `compression-v*` tag is pushed.

## Artifacts

| Artifact | CPU/GPU floor | Linear algebra | Contents |
| --- | --- | --- | --- |
| `plink2-linux-x86_64-v3-mkl` | AVX2/BMI2/FMA | statically linked oneMKL ILP64 | `plink2`, `pgen_compress`, `pgen_rans`, and `libpgen_rans.a` |
| `plink2-macos-arm64-accelerate` | Apple Silicon | Apple Accelerate | same CPU tools and library |
| `plink2-macos-x86_64-v3-accelerate` | AVX2/BMI2/FMA | Apple Accelerate | same CPU tools and library |
| `pgen-rans-cuda-sm75-sm80` | NVIDIA T4 or A100 | not applicable | CUDA decoder benchmark |

Each CPU artifact also contains the public conditional-rANS headers, the
project license, build metadata, and SHA-256 checksums. The Linux PLINK binary
uses static oneMKL libraries, so oneAPI is not required at runtime. System
glibc, libstdc++, libgomp, zlib, and pthread remain dynamic dependencies.

The CUDA artifact is a fat binary containing native `sm_75` and `sm_80`
device images. A compatible NVIDIA driver is still required at runtime.

## Run the workflow

After pushing the branch, open GitHub Actions, select
`Build compression artifacts`, and choose `Run workflow`. A tagged build can
instead be created by pushing a tag such as `compression-v0.1`.

The workflow performs these checks before uploading:

- conditional-rANS codec and container tests;
- `plink2 --version` and `pgen_rans --help` smoke tests;
- expected AVX2/MKL or Accelerate build identification;
- absence of dynamic oneMKL dependencies in the Linux artifact;
- presence of both `sm_75` and `sm_80` CUDA images.

## Local builds

The normal portable Linux/macOS build remains:

```sh
make -C build_dynamic
```

from the `2.0` directory. Linux uses the configured OpenBLAS/LAPACK libraries;
macOS uses Accelerate.

For an AVX2 Linux MKL build, install oneMKL development files, source the
oneAPI environment, and invoke:

```sh
make -C build_dynamic clean
make -C build_dynamic -j"$(nproc)" \
  NO_AVX2= \
  NO_SSE42= \
  DYNAMIC_MKL=1 \
  MKLROOT="$MKLROOT" \
  MKL_IOMP5_DIR="${CMPLR_ROOT}/lib"
```

This local command uses dynamic oneMKL, matching the existing Makefile option.
The artifact workflow supplies a static oneMKL link line for portability.

Build and test the CPU conditional-rANS tools independently with:

```sh
make -f Makefile.pgen_rans test pgen_rans libpgen_rans
```

Build the T4/A100 CUDA fat binary with:

```sh
make -f Makefile.pgen_rans_cuda CUDA_ARCHS="75 80"
```

oneMKL affects PLINK's dense matrix operations; it does not accelerate the
conditional-rANS encoder or decoder. Those tools instead benefit from CPU
instruction targeting, worker count, memory bandwidth, and storage bandwidth.
