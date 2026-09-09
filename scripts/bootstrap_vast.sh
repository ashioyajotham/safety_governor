#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/bootstrap_vast.sh <immutable-git-commit> [vast_bf16|vast_fp16]" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be exported before bootstrap" >&2
  exit 2
fi
if [[ -z "${VAST_IMAGE:-${CONTAINER_IMAGE:-}}" ]]; then
  echo "VAST_IMAGE must name the exact Vast template/base image" >&2
  exit 2
fi

readonly GIT_COMMIT="$1"
readonly PROFILE_NAME="${2:-vast_bf16}"
if [[ "${PROFILE_NAME}" != "vast_bf16" && "${PROFILE_NAME}" != "vast_fp16" ]]; then
  echo "runtime profile must be vast_bf16 or vast_fp16" >&2
  exit 2
fi
readonly VOLUME_ROOT="/data/safety_governor"
readonly REPOSITORY_URL="${SAFETY_GOVERNOR_REPOSITORY_URL:-https://github.com/ashioyajotham/safety_governor.git}"
readonly REPOSITORY_ROOT="${VOLUME_ROOT}/repository"
readonly ENVIRONMENT_ROOT="${VOLUME_ROOT}/venv"
readonly PYTHON_BIN="${PYTHON311_BIN:-python3.11}"
readonly TORCH_VERSION="2.8.0"
readonly PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.11 is required; set PYTHON311_BIN to its executable path" >&2
  exit 2
fi

mkdir -p "${VOLUME_ROOT}/artifacts" "${VOLUME_ROOT}/exports" "${VOLUME_ROOT}/huggingface"
if [[ -d "${REPOSITORY_ROOT}/.git" ]]; then
  git -C "${REPOSITORY_ROOT}" fetch origin
else
  git clone "${REPOSITORY_URL}" "${REPOSITORY_ROOT}"
fi
git -C "${REPOSITORY_ROOT}" checkout --detach "${GIT_COMMIT}"
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain)" ]]; then
  echo "refusing to qualify a dirty checkout" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv --clear "${ENVIRONMENT_ROOT}"
"${ENVIRONMENT_ROOT}/bin/python" -m pip install --upgrade pip
"${ENVIRONMENT_ROOT}/bin/python" -m pip install \
  "torch==${TORCH_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"
"${ENVIRONMENT_ROOT}/bin/python" -m pip install -r "${REPOSITORY_ROOT}/requirements.txt"
cd "${REPOSITORY_ROOT}"
export HF_HOME="${VOLUME_ROOT}/huggingface"
"${ENVIRONMENT_ROOT}/bin/python" -m pip freeze > "${VOLUME_ROOT}/qualified-requirements.txt"
"${ENVIRONMENT_ROOT}/bin/python" -m scripts.validate_dataset datasets/frozen/english_contrastive.jsonl
"${ENVIRONMENT_ROOT}/bin/python" -m scripts.stage1_preflight \
  configs/llama3_8b.yaml \
  --runtime-profile "configs/runtime/${PROFILE_NAME}.yaml" \
  --check-model-access

echo "Vast qualification complete."
echo "Repository: ${REPOSITORY_ROOT}"
echo "Environment: ${ENVIRONMENT_ROOT}"
echo "Artifacts: ${VOLUME_ROOT}/artifacts"
