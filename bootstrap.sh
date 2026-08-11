#!/usr/bin/env bash
# bootstrap.sh — create an isolated venv (sharing the system PyTorch) and build
# pinned FlashInfer + FlashMLA for the H800 (SM90) Sparse MLA benchmark.
# Idempotent: an already-correct install is verified, not rebuilt.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASHMLA_DIR="${PROJECT_DIR}/third_party/FlashMLA"
FLASHMLA_COMMIT="15f13e5030374295491c5ce31b02d7e63a7772c6"
FLASHINFER_VERSION="0.6.11.post1"
TVM_FFI_VERSION="0.1.13.post0"
CUTLASS_DSL_VERSION="4.6.1"
VENV_DIR="${PROJECT_DIR}/.venv"
WHEELS_DIR="${PROJECT_DIR}/assets/wheels"

SYSTEM_PYTHON="${SYSTEM_PYTHON:-/opt/conda/bin/python}"

# /root/.pip/pip.conf forces `user=true`, which breaks installs inside a venv.
export PIP_USER=0

echo "==> Checking system python / torch"
"${SYSTEM_PYTHON}" - <<'EOF'
import sys, torch
assert torch.__version__.startswith("2.9.1+cu130"), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
cap = torch.cuda.get_device_capability(0)
assert cap == (9, 0), f"expected SM90 GPU, got {cap}"
print("system python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
EOF

echo "==> Creating venv (system-site-packages: shares system PyTorch)"
"${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
PIP="${VENV_DIR}/bin/pip"

echo "==> Installing benchmark dependencies"
"${PIP}" install --upgrade pip
"${PIP}" install \
    "flashinfer-python==${FLASHINFER_VERSION}" \
    "apache-tvm-ffi==${TVM_FFI_VERSION}" \
    "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
    "ninja==1.13.0" \
    "matplotlib==3.11.1"

echo "==> Archiving pinned FlashInfer wheel (for environment.json hash)"
mkdir -p "${WHEELS_DIR}"
if ! ls "${WHEELS_DIR}"/flashinfer_python-*.whl >/dev/null 2>&1; then
    "${PIP}" download --no-deps --dest "${WHEELS_DIR}" \
        "flashinfer-python==${FLASHINFER_VERSION}"
fi
sha256sum "${WHEELS_DIR}"/*.whl

echo "==> Verifying FlashMLA checkout"
if [[ ! -d "${FLASHMLA_DIR}/.git" ]]; then
    git clone https://github.com/deepseek-ai/FlashMLA.git "${FLASHMLA_DIR}"
fi
actual_commit="$(git -C "${FLASHMLA_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${FLASHMLA_COMMIT}" ]]; then
    git -C "${FLASHMLA_DIR}" fetch origin "${FLASHMLA_COMMIT}"
    git -C "${FLASHMLA_DIR}" checkout --detach "${FLASHMLA_COMMIT}"
fi
git -C "${FLASHMLA_DIR}" submodule update --init --recursive

if [[ "${FLASHMLA_SKIP_BUILD:-0}" != "1" ]] && \
   ! "${VENV_DIR}/bin/python" -c "import flash_mla" >/dev/null 2>&1; then
    echo "==> Building FlashMLA (SM90 only)"
    cd "${FLASHMLA_DIR}"
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export FLASH_MLA_DISABLE_SM100=1
    export TORCH_CUDA_ARCH_LIST="9.0a"
    export MAX_JOBS=16
    export NVCC_THREADS=4
    # CUDA 13 ships CCCL (libcu++) under include/cccl instead of include/ —
    # make it visible to both the host compiler and nvcc.
    export CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    export NVCC_PREPEND_FLAGS="-I/usr/local/cuda-13.0/include/cccl ${NVCC_PREPEND_FLAGS:-}"
    "${PIP}" install --no-build-isolation -v .
    cd "${PROJECT_DIR}"
else
    echo "==> flash_mla already importable, skipping rebuild"
fi

echo "==> Verifying build"
cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" - <<EOF
import subprocess, torch, flash_mla, flashinfer, tvm_ffi
from sglang_q8kv8 import source_manifest
commit = subprocess.check_output(
    ["git", "-C", "${FLASHMLA_DIR}", "rev-parse", "HEAD"], text=True).strip()
assert commit == "${FLASHMLA_COMMIT}", commit
assert flash_mla.__version__.startswith("1.0.0"), flash_mla.__version__
assert flashinfer.__version__ == "${FLASHINFER_VERSION}", flashinfer.__version__
assert tvm_ffi.__version__ == "${TVM_FFI_VERSION}", tvm_ffi.__version__
manifest = source_manifest()
assert manifest["sglang_commit"] == "5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c"
cap = torch.cuda.get_device_capability(0)
assert cap == (9, 0), cap
# SM90 import + tiny dense-decode smoke
sched, _ = flash_mla.get_mla_metadata()
q = torch.randn(2, 1, 64, 576, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(16, 64, 1, 576, dtype=torch.bfloat16, device="cuda")
bt = torch.arange(16, dtype=torch.int32, device="cuda").view(2, 8)
sl = torch.full((2,), 512, dtype=torch.int32, device="cuda")
out, lse = flash_mla.flash_mla_with_kvcache(q, kv, bt, sl, 512, sched, None, causal=True)
assert out.shape == (2, 1, 64, 512)
torch.cuda.synchronize()
print("flash_mla:", flash_mla.__version__, "commit:", commit)
print("flashinfer:", flashinfer.__version__)
print("sglang q8kv8 source:", manifest["sglang_commit"],
      len(manifest["source_git_blob_sha1"]), "verified files")
print("dense decode smoke OK:", tuple(out.shape))
EOF

echo "==> bootstrap OK"
