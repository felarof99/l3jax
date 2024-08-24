#!/usr/bin/env python
# coding: utf-8

import os
import sys

import flax.training

# Add the current directory to the Python path
sys.path.append("")

from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from huggingface_hub import snapshot_download
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

# Import necessary modules
from . import checkpoint_lib, llama_model, setup, training_pipeline, utils

# Setup environment
setup.setup_environment()

# Import required libraries
globals().update(setup.setup_imports())

# Model configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
JAX_MODEL_NAME = "felafax/llama-3.1-8B-JAX"
model_ckpt_path = "/mnt/persistent-disk/fax/llama3.1_8b_serialized.flax"

# HuggingFace credentials
HUGGINGFACE_USERNAME = input(
    "INPUT: Please provide your HUGGINGFACE_USERNAME: ")
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")

# Load config and tokenizer
config = AutoConfig.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Download model
model_ckpt_path = snapshot_download(repo_id=JAX_MODEL_NAME,
                                    token=HUGGINGFACE_TOKEN)


def get_dataset(*, tokenizer, batch_size=1, seq_length=32, max_examples=None):
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""

    EOS_TOKEN = tokenizer.eos_token

    # Defines formatting function.
    def _format_prompts(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_length + 1,
        )
        return {
            "input_tokens":
            [input_id[:-1] for input_id in tokenized["input_ids"]],
            "target_tokens":
            [input_id[1:] for input_id in tokenized["input_ids"]],
            "loss_masks":
            [input_id[1:] for input_id in tokenized["attention_mask"]],
        }

    def _custom_collate_fn(
            batch: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """
        Collates batch items and converts PyTorch tensors to JAX arrays.
        Applies default_data_collator, then converts tensors to JAX format.
        """
        collated = default_data_collator(batch)
        jax_batch = {}
        for key, value in collated.items():
            jax_batch[key] = (jnp.array(value.numpy()) if isinstance(
                value, torch.Tensor) else value)

        return jax_batch

    # Load and preprocess the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if max_examples:
        dataset = dataset.select(range(max_examples))
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    for split in ["train", "test"]:
        ds[split] = ds[split].map(_tokenize,
                                  batched=True,
                                  remove_columns=dataset.column_names)

    # Create DataLoaders
    dataloader_args = dict(shuffle=True,
                           batch_size=batch_size,
                           collate_fn=_custom_collate_fn)
    train_dataloader = DataLoader(ds["train"], **dataloader_args)
    test_dataloader = DataLoader(ds["test"], **dataloader_args)

    return train_dataloader, test_dataloader


def test_dataset_pipeline(tokenizer):
    """Print shapes of first batch to verify dataset pipeline."""
    train_loader, _ = get_dataset(tokenizer=tokenizer,
                                  batch_size=1,
                                  seq_length=32,
                                  max_examples=512)
    batch = next(iter(train_loader))
    print("Input tokens shape:", batch["input_tokens"].shape)
    print("Target mask shape:", batch["target_tokens"].shape)


test_dataset_pipeline(tokenizer)


@chex.dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int | None = 5
    batch_size: int = 32
    seq_length: int = 64
    dataset_size_limit: int | None = 512
    print_every_n_steps: int = 1


training_cfg = TrainingConfig()

# Configure mesh
devices = jax.devices()
device_count = len(devices)
device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
mesh = Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))

# Initialize model and optimizer
llama_config = llama_model.LlamaConfig.get_standard_llama_config("llama3_8b")
llama_config = llama_model.LlamaConfig.finalize_config(llama_config)
model = llama_model.CausalLlamaModule(
    llama_config,
    dtype=jnp.float32,
    param_dtype=jnp.float32,
)
optimizer = optax.sgd(training_cfg.learning_rate)

# Prepare dataset
train_dataloader, val_dataloader = get_dataset(
    tokenizer=tokenizer,
    seq_length=training_cfg.seq_length,
    max_examples=training_cfg.dataset_size_limit,
)

# Update model path
model_ckpt_path = os.path.join(model_ckpt_path, "llama3.1_8b_serialized.flax")

# Initialize the Trainer
trainer = training_pipeline.Trainer(
    model=model,
    model_ckpt_path=model_ckpt_path,
    model_config=llama_config,
    optimizer=optimizer,
    training_config=training_cfg,
    mesh=mesh,
)

# Train the model
state = trainer.train(mesh, trainer.train_state, train_dataloader)

# Save the trained model
trainer.save_model(state, trainer.gather_fns)
