#!/bin/bash
# run_multihost_test.sh

# Activate venv
source /sglang-jax/.venv/bin/activate

# Set JAX env vars
export JAX_PROCESS_COUNT=2

if [ -z "$JAX_PROCESS_ID" ]; then
    echo "Error: JAX_PROCESS_ID is not set"
    exit 1
fi

export NUM_LAYERS=2

# Set LIBTPU_INIT_ARGS with bypass flag
export LIBTPU_INIT_ARGS="--xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_offload_gather_to_sparsecore=true --xla_tpu_dvfs_p_state=7 --xla_tpu_disable_sparse_core_collective_offload_remover=true --xla_tpu_scoped_vmem_limit_kib=60000 --bypass_vbar_control_service=true"

echo "Running test on Host $JAX_PROCESS_ID with bypass_vbar..."
python3 -u /sglang-jax/python/sgl_jax/test/kernels/test_kimi_int4_loading.py
