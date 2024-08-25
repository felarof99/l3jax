# setup.py
import os


def setup_environment():
    os.environ['HF_HUB_CACHE'] = '/mnt/persistent-disk/hf/'
    os.environ['HF_HOME'] = '/mnt/persistent-disk/hf/'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Note: The following shell commands won't work directly in Python
    # We'll use os.system to execute them
    os.system('export HF_HUB_CACHE="/mnt/persistent-disk/hf/"')
    os.system('export HF_HOME="/mnt/persistent-disk/hf/"')
    os.system('export TOKENIZERS_PARALLELISM=false')


def setup_imports():
    # Standard library imports
    import enum
    import functools
    import os
    import pdb
    import re
    import string
    from dataclasses import dataclass
    from functools import partial
    from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple,
                        Union)

    import chex
    import flax
    import flax.linen as nn
    # JAX and related libraries (including Flax and Optax)
    import jax
    import jax.numpy as jnp
    import lorax
    import optax
    import torch
    from datasets import Dataset, concatenate_datasets, load_dataset
    from flax.core.meta import unbox
    from flax.training.train_state import TrainState
    from jax.experimental import mesh_utils
    from jax.lax import with_sharding_constraint
    # JAX model partitioning and sharding
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as PS
    # Hugging Face Transformers and Datasets
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              default_data_collator)

    # Return a dictionary of all imported modules
    return locals()
