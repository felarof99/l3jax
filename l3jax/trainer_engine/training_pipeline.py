# Standard library imports
import os
import pdb
import enum
import re
import string
from dataclasses import dataclass
import functools
from functools import partial
from typing import Any, List, Dict, Tuple, Optional, Union, Sequence, Mapping

# Third-party imports
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# JAX and related libraries
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from flax.core.meta import unbox
from flax.serialization import from_bytes, to_bytes, to_state_dict, from_state_dict
from flax.training.train_state import TrainState
import optax

# JAX model partitioning and sharding
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

# Local imports

from . import utils
from .utils import cross_entropy_loss_and_accuracy
from . import llama_model
from . import checkpoint_lib 

# Less common imports
import torch

# create a base abstract class for the model called FelafaxModule. It should contain functions like setup, train step, eval step , svae checkpoint and load checkpoint and compute loss.
from abc import ABC, abstractmethod
class FelafaxModule(nn.Module, ABC):
    @abstractmethod
    def setup(self):
        raise NotImplementedError()
    
    @abstractmethod
    def train_step(self, state, rng, batch):
        raise NotImplementedError()
    
    @abstractmethod
    def eval_step(self, state, batch):
        raise NotImplementedError()
    
    @abstractmethod
    def compute_loss(self, state, batch):
        raise NotImplementedError()
    
    @abstractmethod
    def save_checkpoint(self, state, path, checkpointer):
        raise NotImplementedError()
    
    @abstractmethod
    def load_checkpoint(self, path, seq_length, checkpointer):
        raise NotImplementedError()
    
# now create llama model class that inherits from FelafaxModule

class LlamaModel(FelafaxModule):
    def setup(self):
        self.config = llama_model.LlamaConfig()
        self.model = llama_model.LlamaModel(self.config)
        self.optimizer = optax.adam(learning_rate=1e-4)
    
    def _init_fn(self, rng, seq_length):
        rng_generator = utils.NextRNG(rng)
        params = self.model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
        )
        return TrainState.create(params=params, tx=self.optimizer, apply_fn=self.model.apply)
    
    def _get_state_shapes(self, seq_length):
        return jax.eval_shape(
            functools.partial(
                self._init_fn,
                rng=jax.random.PRNGKey(0),
                seq_length=seq_length,
            )
        )
    
    def train_step(self, state, rng, batch):
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
                logits, batch["target_tokens"], batch["loss_masks"]
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return state, rng_generator(), metrics
    
    def eval_step(self, state, batch):
        logits = state.apply_fn(
            state.params,
            batch["input_tokens"],
            deterministic=True,
        ).logits
        return utils.cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )
    
    def compute_loss(self, state, batch):
        return self.eval_step(state, batch)[0]  # Return only the loss
    
    def save_checkpoint(self, state, path, checkpointer):
        checkpointer.save_trainstate_checkpoint(state, path)
    
    def load_checkpoint(self, path, seq_length, checkpointer):
        state_shapes = self._get_state_shapes(seq_length)
        state_shapes_partitioned = utils.match_partition_rules(
            llama_model.LlamaConfig.get_partition_rules(), state_shapes
        )
        shard_fns, _ = utils.make_shard_and_gather_fns(
            state_shapes_partitioned, state_shapes
        )
        state, restored_params = checkpointer.load_trainstate_checkpoint(
            path, state_shapes, shard_fns
        )
        if restored_params is not None:
            state = create_trainstate_from_params(
                restored_params, self.model.apply, self.optimizer
            )
        return state


def init_fn(rng, model, seq_length, optimizer):
    rng_generator = utils.NextRNG(rng)
    params = model.init(
        input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
        position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
        attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
        rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
    )
    return TrainState.create(params=params, tx=optimizer, apply_fn=model.apply)


def get_state_shapes(model, seq_length, optimizer):
    return jax.eval_shape(
        functools.partial(
            init_fn,
            rng=jax.random.PRNGKey(0),
            model=model,
            seq_length=seq_length,
            optimizer=optimizer,
        )
    )


def create_trainstate_from_params(params, model_apply_fn, optimizer):
    return TrainState.create(params=params, apply_fn=model_apply_fn, tx=optimizer)


def get_sharded_create_trainstate_from_params(state_partitioned):
    return pjit(
        create_trainstate_from_params,
        in_shardings=(state_partitioned.params,),
        out_shardings=state_partitioned,
        static_argnums=(1, 2),
        # donate_argnums=(0, ),
    )


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
            logits, batch["target_tokens"], batch["loss_masks"]
        )

    grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = dict(
        loss=loss,
        accuracy=accuracy,
    )
    return state, rng_generator(), metrics


def get_sharded_train_step(state_partitioned):
    return pjit(
        functools.partial(train_step),
        in_shardings=(state_partitioned, PS(), PS()),
        out_shardings=(state_partitioned, PS(), PS()),
        donate_argnums=(0, 1),
    )

def train_loop(
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    train_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    training_cfg: Any,
    mesh: Mesh,
    model_path: str,
    checkpointer: checkpoint_lib.Checkpointer,
) -> train_state.TrainState:
    # initalizes rng generator in utils
    utils.init_rng(99)
    utils.next_rng()

    devices = jax.devices()
    device_count = len(devices)
    device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
    mesh = Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))

    state_shapes = get_state_shapes(model, training_cfg.max_length, optimizer)

    state_shapes_partitioned = utils.match_partition_rules(
        llama_model.LlamaConfig.get_partition_rules(), state_shapes
    )

    shard_fns, gather_fns = utils.make_shard_and_gather_fns(
        state_shapes_partitioned, state_shapes
    )

    sharded_train_step = get_sharded_train_step(state_shapes_partitioned)
    sharded_create_trainstate_from_params = get_sharded_create_trainstate_from_params(
        state_shapes_partitioned
    )

    with mesh:
        state, restored_params = None, None
        
        print("Loading llama JAX model...")
        state, restored_params = checkpointer.load_trainstate_checkpoint(
            "flax_params::" + model_path, state_shapes, shard_fns
        )
        if restored_params is not None:
            state = sharded_create_trainstate_from_params(
                restored_params, model.apply, optimizer
            )
            del restored_params
        else:
            raise ValueError("Failed to load checkpoint")

        for epoch in range(training_cfg.num_epochs):
            print(f"Starting epoch {epoch} of training...")
            
            for step, train_batch in enumerate(train_dataloader):
                # Place the batch on the appropriate devices
                train_batch = jax.device_put(train_batch, NamedSharding(mesh, PS()))

                sharded_rng = utils.next_rng()

                # Perform a single training step
                state, sharded_rng, metrics = sharded_train_step(
                    state, sharded_rng, train_batch
                )

                if step % training_cfg.print_every_n_steps == 0:
                    print(
                        f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                    )

                if training_cfg.max_steps and step >= training_cfg.max_steps:
                    break
        return state, gather_fns
