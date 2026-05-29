import logging  
import numpy as np  
import jax  
import jax.numpy as jnp  
from flax import nnx  
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  
  
from sgl_jax.srt.configs.model_config import ModelConfig  
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend  
from sgl_jax.srt.layers.logits_processor import LogitsMetadata  
from sgl_jax.srt.managers.schedule_batch import ForwardMode, ModelWorkerBatch  
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool, MemoryPools  
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch  
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_vl_generation import (  
    KimiK25ForConditionalGeneration,  
)  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
MODEL_PATH = "/path/to/kimi-k2.5"  
PAGE_SIZE = 128   # MLAAttentionBackend asserts page_size > 1  
  
  
def main():  
    # ── 1. Mesh ──────────────────────────────────────────────────────────────  
    devices = jax.devices()  
    n = len(devices)  
    mesh = Mesh(  
        np.array(devices).reshape(1, n),  
        axis_names=("data", "tensor"),  
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),  
    )  
  
    # ── 2. ModelConfig ───────────────────────────────────────────────────────  
    model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)  
    text_config = model_config.hf_text_config  
    text_config.quantization_config = None   # avoid dict-vs-object error  
  
    # ── 3. Create model (eval_shape) + load weights ──────────────────────────  
    with jax.sharding.use_mesh(mesh):  
        model = nnx.eval_shape(  
            lambda: KimiK25ForConditionalGeneration(  
                model_config.hf_config, mesh=mesh, dtype=jnp.bfloat16  
            )  
        )  
        model.load_weights(model_config)  
    logger.info("Weights loaded.")  
  
    # ── 4. KV pool + MemoryPools ─────────────────────────────────────────────  
    # Keep pool small: 10 pages × PAGE_SIZE tokens each  
    pool_size = PAGE_SIZE * 10  
    with jax.sharding.use_mesh(mesh):  
        kv_pool = MLATokenToKVPool(  
            size=pool_size,  
            page_size=PAGE_SIZE,  
            dtype=jnp.bfloat16,  
            kv_lora_rank=text_config.kv_lora_rank,       # 512  
            qk_rope_head_dim=text_config.qk_rope_head_dim,  # 64  
            layer_num=text_config.num_hidden_layers,      # 61  
            mesh=mesh,  
        )  
    memory_pools = MemoryPools(token_to_kv_pool=kv_pool)  
  
    # ── 5. Attention backend ─────────────────────────────────────────────────  
    attn_backend = MLAAttentionBackend(  
        num_attn_heads=text_config.num_attention_heads,   # 128  
        kv_lora_rank=text_config.kv_lora_rank,            # 512  
        qk_nope_head_dim=text_config.qk_nope_head_dim,    # 128  
        qk_rope_head_dim=text_config.qk_rope_head_dim,    # 64  
        v_head_dim=text_config.v_head_dim,                # 128  
        page_size=PAGE_SIZE,  
        mesh=mesh,  
        attention_data_partition_axis="data",  
    )  
  
    # ── 6. Dummy batch data ──────────────────────────────────────────────────  
    seq_len = 10  
    bs = 1  
  
    input_ids_np      = np.arange(1, seq_len + 1, dtype=np.int32)  
    positions_np      = np.arange(seq_len, dtype=np.int32)  
    seq_lens_np       = np.array([seq_len], dtype=np.int32)  
    extend_seq_lens   = np.array([seq_len], dtype=np.int32)  
    extend_prefix_lens = np.zeros(bs, dtype=np.int32)  
    req_pool_indices  = np.zeros(bs, dtype=np.int32)  
  
    # One full page of cache slots (PAGE_SIZE slots, first seq_len are "real")  
    cache_loc    = np.arange(PAGE_SIZE, dtype=np.int32)  
    out_cache_loc = np.arange(seq_len, dtype=np.int32)  
  
    # ── 7. ModelWorkerBatch (needed by get_forward_metadata) ─────────────────  
    mwb = ModelWorkerBatch(  
        bid=1,  
        forward_mode=ForwardMode.EXTEND,  
        input_ids=input_ids_np,  
        real_input_ids_len=seq_len,  
        seq_lens=seq_lens_np,  
        out_cache_loc=out_cache_loc,  
        req_pool_indices=req_pool_indices,  
        positions=positions_np,  
        cache_loc=cache_loc,  
        extend_seq_lens=extend_seq_lens,  
        extend_prefix_lens=extend_prefix_lens,  
        sampling_info=None,  
        return_logprob=False,  
        return_output_logprob_only=False,  
        top_logprobs_nums=None,  
        token_ids_logprobs=None,  
        extend_logprob_start_lens=None,  
        extend_input_logprob_token_ids=None,  
        logits_indices=extend_seq_lens,  
        real_bs=bs,  
        real_bs_per_dp=[bs],  
        dp_size=1,  
        per_dp_bs_size=bs,  
        lora_ids=["0"] * bs,  
        spec_info=None,  
    )  
  
    # ── 8. ForwardBatch ──────────────────────────────────────────────────────  
    forward_metadata = attn_backend.get_forward_metadata(mwb)  
    attn_backend.forward_metadata = forward_metadata  
  
    fb = ForwardBatch(  
        bid=1,  
        forward_mode=ForwardMode.EXTEND,  
        batch_size=bs,  
        input_ids=jnp.array(input_ids_np),  
        req_pool_indices=jnp.array(req_pool_indices),  
        seq_lens=jnp.array(seq_lens_np),  
        out_cache_loc=jnp.array(out_cache_loc),  
        positions=jnp.array(positions_np),  
        attn_backend=attn_backend,  
        cache_loc=jnp.array(cache_loc),  
        extend_prefix_lens=jnp.array(extend_prefix_lens),  
        extend_seq_lens=jnp.array(extend_seq_lens),  
        capture_hidden_mode=CaptureHiddenMode.NULL,  
    )  
  
    # ── 9. LogitsMetadata ────────────────────────────────────────────────────  
    logits_metadata = LogitsMetadata(  
        forward_mode=ForwardMode.EXTEND,  
        capture_hidden_mode=CaptureHiddenMode.NULL,  
    )  
  
    # ── 10. Forward pass ─────────────────────────────────────────────────────  
    logger.info("Running forward pass...")  
    with jax.sharding.use_mesh(mesh):  
        output, layers_kv_fused, _, layers_topk_ids = model(  
            fb, memory_pools, logits_metadata  
        )  
  
    logits = output.next_token_logits  
    logger.info("Forward pass OK. logits shape: %s", logits.shape)  
    assert logits.shape[-1] == text_config.vocab_size, (  
        f"Expected vocab_size={text_config.vocab_size}, got {logits.shape[-1]}"  
    )  
    logger.info("Test passed!")  
  
  
if __name__ == "__main__":  
    main()
