# Standard library imports
import enum
import functools
import os
import pdb
import re
import string
# create a base abstract class for the model called FelafaxModule. It should contain functions like setup, train step, eval step , svae checkpoint and load checkpoint and compute loss.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import chex
import flax
import flax.linen as nn
# JAX and related libraries
import jax
import jax.numpy as jnp
# Third-party imports
import numpy as np
import optax
# Less common imports
import torch
from flax import struct
from flax.core.meta import unbox
from flax.serialization import (from_bytes, from_state_dict, to_bytes,
                                to_state_dict)
from flax.training import train_state
from flax.training.train_state import TrainState
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
# JAX model partitioning and sharding
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from . import checkpoint_lib, llama_model, utils
from .utils import cross_entropy_loss_and_accuracy


@chex.dataclass(frozen=True)
class FxTrainingConfig:
    """Configures the training pipeline with hyperparameters and limits."""
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int | None = 5
    batch_size: int = 32
    max_length: int = 64
    dataset_size_limit: int | None = 512
    print_every_n_steps: int = 1


class FxTrainState(train_state.TrainState):
    """Stores the training state, including the model and configuration."""
    training_config: FxTrainingConfig


class Trainer:

    def __init__(
        self,
        model,
        model_ckpt_path,
        optimizer,
        training_config,
        mesh,
    ):
        self.model = model
        self.model_ckpt_path = model_ckpt_path

        self.optimizer = optimizer
        self.training_config = training_config
        self.mesh = mesh

        self.checkpointer = checkpoint_lib.Checkpointer(
            checkpoint_lib.Checkpointer.get_default_config(),
            checkpoint_dir=os.path.dirname(model_ckpt_path),
            enable_checkpointer=jax.process_index() == 0,
        )

    @staticmethod
    def init_fn(rng, model, seq_length, optimizer):
        rng_generator = utils.NextRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
        )
        return TrainState.create(params=params,
                                 tx=optimizer,
                                 apply_fn=model.apply)

    def get_state_shapes(self, seq_length):
        return jax.eval_shape(
            functools.partial(
                self.init_fn,
                rng=jax.random.PRNGKey(0),
                model=self.model,
                seq_length=seq_length,
                optimizer=self.optimizer,
            ))

    @staticmethod
    def create_trainstate_from_params(params, model_apply_fn, optimizer):
        return TrainState.create(params=params,
                                 apply_fn=model_apply_fn,
                                 tx=optimizer)

    def get_sharded_create_trainstate_from_params(self, state_partitioned):
        return pjit(
            self.create_trainstate_from_params,
            in_shardings=(state_partitioned.params, ),
            out_shardings=state_partitioned,
            static_argnums=(1, 2),
        )

    @staticmethod
    def train_step(state, rng, batch):
        rng_generator = utils.NextRNG(rng)
        batch = utils.with_sharding_constraint(batch, PS(("dp", "fsdp")))

        def loss_and_accuracy(params):
            logits = state.apply_fn(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
            ).logits
            return utils.cross_entropy_loss_and_accuracy(
                logits, batch["target_tokens"], batch["loss_masks"])

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return state, rng_generator(), metrics

    def get_sharded_train_step(self, state_partitioned):
        return pjit(
            functools.partial(self.train_step),
            in_shardings=(state_partitioned, PS(), PS()),
            out_shardings=(state_partitioned, PS(), PS()),
            donate_argnums=(0, 1),
        )

    def train(self, mesh, train_dataloader):
        utils.init_rng(99)
        utils.next_rng()

        state_shapes = self.get_state_shapes(self.training_config.max_length)
        state_shapes_partitioned = utils.match_partition_rules(
            llama_model.LlamaConfig.get_partition_rules(), state_shapes)
        shard_fns, gather_fns = utils.make_shard_and_gather_fns(
            state_shapes_partitioned, state_shapes)
        sharded_train_step = self.get_sharded_train_step(
            state_shapes_partitioned)
        sharded_create_trainstate_from_params = self.get_sharded_create_trainstate_from_params(
            state_shapes_partitioned)

        with mesh:
            state, restored_params = None, None

            print("Loading llama JAX model...")
            state, restored_params = self.checkpointer.load_trainstate_checkpoint(
                "flax_params::" + self.model_ckpt_path, state_shapes,
                shard_fns)
            if restored_params is not None:
                state = sharded_create_trainstate_from_params(
                    restored_params, self.model.apply, self.optimizer)
                del restored_params
            else:
                raise ValueError("Failed to load checkpoint")

            for epoch in range(self.training_config.num_epochs):
                print(f"Starting epoch {epoch} of training...")

                for step, train_batch in enumerate(train_dataloader):
                    train_batch = jax.device_put(train_batch,
                                                 NamedSharding(mesh, PS()))
                    sharded_rng = utils.next_rng()
                    state, sharded_rng, metrics = sharded_train_step(
                        state, sharded_rng, train_batch)

                    if step % self.training_config.print_every_n_steps == 0:
                        print(
                            f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                        )

                    if self.training_config.max_steps and step >= self.training_config.max_steps:
                        break

        return state, gather_fns

    def save_model(self, state, gather_fns):
        self.checkpointer.save_train_state_to_file(
            train_state=state,
            gather_fns=gather_fns,
            path=os.path.join(self.model_ckpt_path, "trained_llama.flax"),
        )
