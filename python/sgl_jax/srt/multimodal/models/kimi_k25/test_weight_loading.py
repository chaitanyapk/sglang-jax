import logging
import os

import jax

# Set up logging immediately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_weight_loading")

# Initialize JAX distributed as early as possible to avoid backend init issues
if "JAX_COORDINATOR_ADDRESS" in os.environ:
    logger.info("Initializing JAX distributed at startup...")
    try:
        jax.distributed.initialize()
        logger.info("JAX Distributed Initialized successfully at startup.")

        # Set TPU overrides after JAX init but before backend init
        process_id = int(os.environ.get("JAX_PROCESS_ID", 0))
        host0_name = "gke-tpu-4a99f854-2zmz"
        host1_name = "gke-tpu-4a99f854-ptwl"

        if process_id == 0:
            os.environ["TPU_WORKER_HOSTNAMES"] = host0_name
            os.environ["TPU_PROCESS_ADDRESSES"] = f"{host0_name}:8471"
            os.environ["TPU_HOSTNAME_OVERRIDE"] = host0_name
            os.environ["MEGASCALE_SLICE_ID"] = "0"
        elif process_id == 1:
            os.environ["TPU_WORKER_HOSTNAMES"] = host1_name
            os.environ["TPU_PROCESS_ADDRESSES"] = f"{host1_name}:8471"
            os.environ["TPU_HOSTNAME_OVERRIDE"] = host1_name
            os.environ["MEGASCALE_SLICE_ID"] = "1"

        os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
        os.environ["MEGASCALE_NUM_SLICES"] = "2"
        os.environ["MEGASCALE_COORDINATOR_ADDRESS"] = f"{host0_name}:9915"

        logger.info(
            "TPU overrides set in Python: TPU_WORKER_HOSTNAMES=%s, TPU_HOSTNAME_OVERRIDE=%s, MEGASCALE_SLICE_ID=%s, TPU_HOST_BOUNDS=%s",
            os.environ["TPU_WORKER_HOSTNAMES"],
            os.environ["TPU_HOSTNAME_OVERRIDE"],
            os.environ["MEGASCALE_SLICE_ID"],
            os.environ["TPU_HOST_BOUNDS"],
        )
    except Exception as e:
        logger.info("JAX distributed init failed or already initialized: %s", e)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (  # noqa: E402
    KimiK25ModelVitConfig,
)
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import (  # noqa: E402
    Kimi_K25_VisionModel,
)

model_path = "/dsk/models/kimi_original/"

# 1. Build mesh - jax.devices() returns TPU cores on a TPU machine
devices = jax.devices()
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))

# 2. Load config
config = KimiK25ModelVitConfig()
config.model_path = model_path
config.model_class = Kimi_K25_VisionModel

# 3. Create model structure and allocate memory on TPU
with jax.set_mesh(mesh):
    model = Kimi_K25_VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)

# 4. Sample params before loading
before = model.vision_tower.encoder.blocks[0].attn.qkv_proj.kernel.value.mean().item()
print(f"Before weight loading, blocks[0].attn.qkv_proj.kernel mean: {before}")

# 5. Load weights - reads on CPU, shards to TPU
model.load_weights(config)

# 6. Verify values changed
after = model.vision_tower.encoder.blocks[0].attn.qkv_proj.kernel.value.mean().item()
print(f"After weight loading, blocks[0].attn.qkv_proj.kernel mean: {after}")
assert before != after, "Weights did not change!"
print("SUCCESS: Weights successfully loaded and verified changed!")
