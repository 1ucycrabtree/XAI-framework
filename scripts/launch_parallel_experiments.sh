#!/usr/bin/env bash
set -euo pipefail

# Launch all KernelSHAP + TabularLIME experiment configs in parallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/xai_framework:${PYTHONPATH:-}"

KERNEL_CONFIGS=(
  "KernelSHAP_FP_Drift.yaml"
  "KernelSHAP_FP_K.yaml"
  "KernelSHAP_FP_Noise.yaml"
  "KernelSHAP_TP_Drift.yaml"
  "KernelSHAP_TP_K.yaml"
  "KernelSHAP_TP_Noise.yaml"
)
KERNEL_LOGS=(
  "KernelSHAP_FP_Drift.log"
  "KernelSHAP_FP_K.log"
  "KernelSHAP_FP_Noise.log"
  "KernelSHAP_TP_Drift.log"
  "KernelSHAP_TP_K.log"
  "KernelSHAP_TP_Noise.log"
)

LIME_CONFIGS=(
  "TabularLIME_FP_Drift.yaml"
  "TabularLIME_FP_K.yaml"
  "TabularLIME_FP_Noise.yaml"
  "TabularLIME_TP_Drift.yaml"
  "TabularLIME_TP_K.yaml"
  "TabularLIME_TP_Noise.yaml"
)
LIME_LOGS=(
  "TabularLIME_FP_Drift.log"
  "TabularLIME_FP_K.log"
  "TabularLIME_FP_Noise.log"
  "TabularLIME_TP_Drift.log"
  "TabularLIME_TP_K.log"
  "TabularLIME_TP_Noise.log"
)

run_job() {
  local core_set="$1"
  local omp_threads="$2"
  local mkl_threads="$3"
  local config_name="$4"
  local log_name="$5"

  if [[ ! -f "config/${config_name}" ]]; then
    echo "Missing config: config/${config_name}" >&2
    return 1
  fi

  taskset -c "${core_set}" \
    env OMP_NUM_THREADS="${omp_threads}" MKL_NUM_THREADS="${mkl_threads}" \
    nohup python -m xai_framework --config "${config_name}" \
      > "logs/${log_name}" 2>&1 &
}

TOTAL_CORES="$(nproc)"
KERNEL_JOB_COUNT="${#KERNEL_CONFIGS[@]}"
LIME_JOB_COUNT="${#LIME_CONFIGS[@]}"
MIN_REQUIRED_CORES=$((KERNEL_JOB_COUNT + LIME_JOB_COUNT))

if (( TOTAL_CORES < MIN_REQUIRED_CORES )); then
  echo "Need at least ${MIN_REQUIRED_CORES} cores for 12 parallel jobs." >&2
  echo "Detected: ${TOTAL_CORES}. Reduce jobs or request more cores." >&2
  exit 1
fi

echo "Detected ${TOTAL_CORES} CPU cores."
echo "Launching KernelSHAP processes (1 core each)..."
for i in "${!KERNEL_CONFIGS[@]}"; do
  core="${i}"
  run_job "${core}" "1" "1" "${KERNEL_CONFIGS[$i]}" "${KERNEL_LOGS[$i]}"
done

echo "KernelSHAP processes launched. Waiting 5 seconds before LIME..."
sleep 5

echo "Launching TabularLIME processes (up to 4 cores each)..."
remaining_cores=$((TOTAL_CORES - KERNEL_JOB_COUNT))
base_allocation=$((remaining_cores / LIME_JOB_COUNT))
extra=$((remaining_cores % LIME_JOB_COUNT))

next_core="${KERNEL_JOB_COUNT}"
for i in "${!LIME_CONFIGS[@]}"; do
  width="${base_allocation}"
  if (( i < extra )); then
    width=$((width + 1))
  fi
  if (( width > 4 )); then
    width=4
  fi
  if (( width < 1 )); then
    width=1
  fi

  start="${next_core}"
  end=$((start + width - 1))
  if (( end >= TOTAL_CORES )); then
    end=$((TOTAL_CORES - 1))
  fi

  if (( start == end )); then
    core_set="${start}"
  else
    core_set="${start}-${end}"
  fi

  run_job "${core_set}" "${width}" "${width}" "${LIME_CONFIGS[$i]}" "${LIME_LOGS[$i]}"
  next_core=$((end + 1))
done

echo "All processes launched."
echo "Use: ps -fu \"${USER}\" | rg xai_framework"
echo "Use: tail -f logs/<name>.log"
