import logging
import os
import jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_jax_init")

logger.info("Initializing JAX distributed...")
jax.distributed.initialize()
logger.info("SUCCESS: JAX distributed initialized!")
