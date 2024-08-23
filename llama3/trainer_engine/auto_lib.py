import functools
import gc
import re
import warnings
import jax
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
from huggingface_hub import snapshot_download


MODEL_NAME_TO_DOWNLOAD_CONFIG = {
    "llama3.1-8b-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B",
        "felafax_model_name": "felafax/llama-3.1-8B-JAX",
    },
}


class AutoJAXModelForCausalLM:

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: str,
        **kwargs,
    ) -> Tuple[str, AutoConfig, AutoTokenizer]:
        """Downloads the model from HF and returns the downloaded model path, config, and tokenizer."""
        download_config = MODEL_NAME_TO_DOWNLOAD_CONFIG[model_name]

        config = AutoConfig.from_pretrained(
            download_config["hf_model_name"], token=huggingface_token
        )

        tokenizer = AutoTokenizer.from_pretrained(
            download_config["hf_model_name"],
            token=huggingface_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_path = snapshot_download(
            repo_id=download_config["felafax_model_name"],
            token=huggingface_token,
        )
        return model_path, config, tokenizer
