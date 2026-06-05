#!/bin/bash
# run_pure_jax_test.sh
source /sglang-jax/.venv/bin/activate
export JAX_PROCESS_COUNT=2
echo "Running pure JAX init test on Host $JAX_PROCESS_ID..."
python3 /sglang-jax/python/sgl_jax/test/kernels/test_jax_init.py
