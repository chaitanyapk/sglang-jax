import logging  
import os  
  
import jax  
import jax.numpy as jnp  
import numpy as np  
from flax import nnx  
  
from sgl_jax.srt.configs.model_config import ModelConfig  
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vl_generation import (  
    KimiK25ForConditionalGeneration,  
)  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
MODEL_PATH = "local/kimi"  # update this  
  
  
def test_weight_loading():  
  
    # 1. Create ModelConfig (reads config.json, sets hf_text_config → text_config)  
    model_config = ModelConfig(  
        model_path=MODEL_PATH,  
        trust_remote_code=True,  
        dtype="bfloat16",  
    )  
    text_config = get_hf_text_config(model_config.hf_config)
    text_config.quantization_config = None   

    logger.info("hf_text_config type: %s", type(model_config.hf_text_config))  
  
    # 2. Create mesh with Explicit axis type  
    devices = jax.devices()  
    logger.info("Available devices: %s", devices)  

    n = len(devices)  # 8 on your TPU  
    mesh = jax.sharding.Mesh(  
        np.array(devices).reshape(1, n),   # data=1, tensor=8  
        ("data", "tensor"),  
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),  
    )
  
    # 3. Create model shape (no real weights yet) inside mesh context  
    with jax.sharding.use_mesh(mesh):  
        model = nnx.eval_shape(  
            lambda: KimiK25ForConditionalGeneration(  
                config=model_config.hf_text_config,  
                mesh=mesh,  
                dtype=jnp.bfloat16,  
            )  
        )  
    logger.info("Model shape created successfully")  
  
    # 4. Load weights (allocates real arrays and fills from safetensors)  
    with jax.sharding.use_mesh(mesh):  
        model.load_weights(model_config)  
    logger.info("Weights loaded successfully")  
  
    # 5. Verify a parameter has real values (not ShapeDtypeStruct)  
    try:  
        val = model.model.embed_tokens.embedding.value  
        logger.info(  
            "embed_tokens shape=%s mean=%.6f", val.shape, float(val.mean())  
        )  
        val2 = model.model.layers[0].self_attn.q_a_proj.weight.value  
        logger.info(  
            "layers[0].q_a_proj shape=%s mean=%.6f", val2.shape, float(val2.mean())  
        )  
    except Exception as e:  
        logger.warning("Could not sample params: %s", e)  
  
  
if __name__ == "__main__":  
    test_weight_loading()
