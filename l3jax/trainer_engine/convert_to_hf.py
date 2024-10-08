import gc
import json
import math
import os
import shutil

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.traverse_util import flatten_dict
from transformers import LlamaConfig, LlamaForCausalLM

from . import config_lib
from .checkpoint_lib import Checkpointer as StreamingCheckpointer
from .llama_model import LlamaConfig as LLaMAConfigurator
from .utils import float_tensor_to_dtype

config = config_lib.config_dict(
    load_checkpoint=
    'trainstate::/mnt/persistent-disk/hf/models--felafax--llama-3.1-8B-JAX/snapshots/ebca17f216e4c02e0f31cc47264a9d65a4f5b9a9/streaming_params',
    output_dir='/mnt/persistent-disk/easy/e2hf/',
    llama=LLaMAConfigurator("llama3_8b"),
)


def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def load_and_convert_checkpoint(flax_params):
    # _, flax_params = StreamingCheckpointer.load_trainstate_checkpoint(path)
    flax_params = flatten_dict(flax_params['params'], sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
            tensor = tensor.T
        torch_params[key] = torch.tensor(float_tensor_to_dtype(tensor, 'fp32'),
                                         dtype=torch.float16)
    return torch_params


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def permute(w, n_heads, input_dim, output_dim):
    # permute for sliced rotary embedding
    return w.view(n_heads, output_dim // n_heads // 2, 2,
                  input_dim).transpose(1, 2).reshape(output_dim, input_dim)


def write_model(loaded, model_path):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    llama_config = LLaMAConfigurator.finalize_config(config.llama)

    n_layers = llama_config.num_hidden_layers
    n_heads = llama_config.num_attention_heads
    n_kv_heads = llama_config.num_key_value_heads
    dim = llama_config.hidden_size
    dims_per_head = dim // n_heads
    base = llama_config.rope_theta
    inv_freq = 1.0 / (base**(torch.arange(0, dims_per_head, 2).float() /
                             dims_per_head))

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight":
            permute(
                loaded[f"transformer.h.{layer_i}.attention.wq.kernel"],
                llama_config.num_attention_heads,
                llama_config.hidden_size,
                llama_config.hidden_size,
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight":
            permute(
                loaded[f"transformer.h.{layer_i}.attention.wk.kernel"],
                llama_config.num_key_value_heads,
                llama_config.hidden_size,
                llama_config.hidden_size //
                (llama_config.num_attention_heads //
                 llama_config.num_key_value_heads),
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight":
            loaded[f"transformer.h.{layer_i}.attention.wv.kernel"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight":
            loaded[f"transformer.h.{layer_i}.attention.wo.kernel"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight":
            loaded[f"transformer.h.{layer_i}.feed_forward.w1.kernel"],
            f"model.layers.{layer_i}.mlp.down_proj.weight":
            loaded[f"transformer.h.{layer_i}.feed_forward.w2.kernel"],
            f"model.layers.{layer_i}.mlp.up_proj.weight":
            loaded[f"transformer.h.{layer_i}.feed_forward.w3.kernel"],
            f"model.layers.{layer_i}.input_layernorm.weight":
            loaded[f"transformer.h.{layer_i}.attention_norm.kernel"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight":
            loaded[f"transformer.h.{layer_i}.ffn_norm.kernel"],
        }

        state_dict[
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    # Unsharded
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.embedding"],
        "model.norm.weight": loaded["transformer.ln_f.kernel"],
        "lm_head.weight": loaded["lm_head.kernel"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict,
               os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        vocab_size=llama_config.vocab_size,
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=llama_config.num_key_value_heads,
        initializer_range=llama_config.initializer_range,
        rms_norm_eps=llama_config.rms_norm_eps,
        max_position_embeddings=llama_config.max_position_embeddings,
        rope_theta=llama_config.rope_theta,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path,
                                             torch_dtype=torch.float16)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)


def main(config, flax_params):
    assert config.load_checkpoint != "" and config.output_dir != ""
    write_model(
        load_and_convert_checkpoint(flax_params),
        model_path=config.output_dir,
    )
