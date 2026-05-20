import jax  
import numpy as np  
from flax import nnx  
import jax.numpy as jnp
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import Kimi_K25_VisionModel  
from sgl_jax.srt.multimodal.configs.config_registry import get_qwen_vl_config  
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization
from sgl_jax.srt.layers.linear import QuantizedLinear, LinearBase

model_path = "/local/moonshotai/Kimi-K2.5"  
quant_config_path = "int4.yaml"  # Loaded from our newly created config
  
# 1. Build mesh — jax.devices() returns TPU cores on a TPU machine  
devices = jax.devices()  
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))  


def run_standard_loading_test():
    print("\n=== Running TEST 1: Standard (Unquantized) Weight Loading ===")
    # 2. Load config
    config = get_qwen_vl_config(model_path)  
    config.model_path = model_path  
    config.model_class = Kimi_K25_VisionModel  
    config.quantization_config = None # Explicitly disable quantization
      
    # 3. Create model structure (no memory allocated yet)  
    with jax.set_mesh(mesh):  
        model = nnx.eval_shape(  
            lambda: Kimi_K25_VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)  
        )  
    
    # Verify it has standard LinearBase structure
    target_block = model.vision_tower.encoder.blocks[0].attn.qkv_proj
    assert isinstance(target_block, LinearBase), "Expected LinearBase layer!"
    
    # 4. Sample params before loading  
    before = target_block.weight[...]
      
    # 5. Load weights — reads on CPU, shards to TPU  
    model.load_weights(config)  
      
    # 6. Verify values changed  
    after = target_block.weight[...]
    assert not jnp.array_equal(before, after), "Weights did not change — check your weight mappings"
    print(f"✅ Standard weight loading verified successfully!")
    print(f"   qkv_proj mean: {after.mean().item():.6f}")


def run_quantized_loading_test():
    print("\n=== Running TEST 2: Quantized (INT4) Weight Loading ===")
    # 2. Load config and inject Quantization Config
    config = get_qwen_vl_config(model_path)  
    config.model_path = model_path  
    config.model_class = Kimi_K25_VisionModel  
    config.quantization_config = QuantizationConfig.from_path(quant_config_path)
      
    # 3. Create model structure (no memory allocated yet)  
    with jax.set_mesh(mesh):  
        model = nnx.eval_shape(  
            lambda: Kimi_K25_VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)  
        )  
        
    # 4. Apply Linear Quantization Walker
    # This will replace all LinearBase layers in Kimi Vision Tower with QuantizedLinear layers
    model = apply_linear_quantization(config, model, is_static_input=True)

    # 5. Verify structures were successfully swapped
    target_block = model.vision_tower.encoder.blocks[0].attn.qkv_proj
    assert isinstance(target_block, QuantizedLinear), "LinearBase layer was not replaced by QuantizedLinear!"
    print("✅ Quantization structure replacement verified successfully!")
    print(f"   qkv_proj.weight_q abstract shape: {target_block.weight_q.value.shape}")
    print(f"   qkv_proj.weight_q abstract dtype: {target_block.weight_q.value.dtype}")

    # 6. Sample abstract params before loading  
    before_shape = target_block.weight_q.value.shape
      
    # 7. Load weights — reads packed file on CPU, unpacks on-the-fly, and streams to dynamic TPU HBM variables
    model.load_weights(config)  
      
    # 8. Verify loaded weights
    # Access concrete value after weight loading has run
    after_val = target_block.weight_q.value
    print("✅ Weights loaded successfully under quantized static branch!")
    print(f"   Loaded weight_q shape: {after_val.shape}")
    print(f"   Loaded weight_q dtype: {after_val.dtype}")
    print(f"   Loaded weight_q mean:  {after_val[...].mean().item():.6f}")


if __name__ == "__main__":
    run_standard_loading_test()
    run_quantized_loading_test()
