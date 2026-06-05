#!/bin/bash
# run_pure_jax_test_with_overrides.sh
source /sglang-jax/.venv/bin/activate
export JAX_PROCESS_COUNT=2
export TPU_WORKER_HOSTNAMES="gke-tpu-4a99f854-2zmz,gke-tpu-4a99f854-ptwl"
export TPU_PROCESS_ADDRESSES="gke-tpu-4a99f854-2zmz:8471,gke-tpu-4a99f854-ptwl:8471"

if [ "$JAX_PROCESS_ID" = "0" ]; then
    export TPU_HOSTNAME_OVERRIDE="gke-tpu-4a99f854-2zmz"
elif [ "$JAX_PROCESS_ID" = "1" ]; then
    export TPU_HOSTNAME_OVERRIDE="gke-tpu-4a99f854-ptwl"
fi
export MEGASCALE_COORDINATOR_ADDRESS="gke-tpu-4a99f854-2zmz:9915"

echo "Running JAX init test with short overrides on Host $JAX_PROCESS_ID..."
python3 -u /sglang-jax/python/sgl_jax/test/kernels/test_jax_init.py
