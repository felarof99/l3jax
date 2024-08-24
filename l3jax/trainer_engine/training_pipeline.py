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
from dataclasses import dataclass
from typing import Any, Tuple
import jax.numpy as jnp
from flax import struct

from flax.training import train_state
import optax

class FelafaxState(train_state.TrainState):
    config: Any
    model: Any

class FelafaxModule(ABC):
    @staticmethod
    @abstractmethod
    def setup(config: Any) -> FelafaxState:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def create_train_state(rng: Any, config: Any) -> FelafaxState:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def train_step(state: FelafaxState, rng: Any, batch: Any) -> Tuple[FelafaxState, Any, dict]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def eval_step(state: FelafaxState, batch: Any) -> Tuple[float, float]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def compute_loss(state: FelafaxState, batch: Any) -> float:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def save_checkpoint(state: FelafaxState, path: str, checkpointer: Any) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_checkpoint(path: str, seq_length: int, checkpointer: Any) -> FelafaxState:
        raise NotImplementedError()
########################################################################
# LlamaModel class implementation
# now create llama model class that inherits from FelafaxModule
########################################################################
class LlamaModel(FelafaxModule):
    @staticmethod
    def setup(config: Any) -> FelafaxState:
        return LlamaModel.create_train_state(jax.random.PRNGKey(0), config)

    @staticmethod
    def create_train_state(rng: Any, config: Any) -> FelafaxState:
        llama_config = llama_model.LlamaConfig()
        model = llama_model.LlamaModel(llama_config)
        params = LlamaModel._init_params(rng, config.seq_length, model)
        tx = optax.adam(learning_rate=config.learning_rate)
        return FelafaxState.create(apply_fn=model.apply, params=params, tx=tx, config=config, model=model)

    @staticmethod
    def _init_params(rng: Any, seq_length: int, model: Any) -> Any:
        rng_generator = utils.NextRNG(rng)
        return model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
        )

    @staticmethod
    def train_step(state: FelafaxState, rng: Any, batch: Any) -> Tuple[FelafaxState, Any, dict]:
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
        new_state = state.apply_gradients(grads=grads)
        metrics = dict(loss=loss, accuracy=accuracy)
        return new_state, rng_generator(), metrics

    @staticmethod
    def eval_step(state: FelafaxState, batch: Any) -> Tuple[float, float]:
        logits = state.apply_fn(
            state.params,
            batch["input_tokens"],
            deterministic=True,
        ).logits
        return utils.cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )

    @staticmethod
    def compute_loss(state: FelafaxState, batch: Any) -> float:
        return LlamaModel.eval_step(state, batch)[0]  # Return only the loss

    @staticmethod
    def save_checkpoint(state: FelafaxState, path: str, checkpointer: Any) -> None:
        checkpointer.save_trainstate_checkpoint(state, path)

    @staticmethod
    def load_checkpoint(path: str, seq_length: int, checkpointer: Any) -> FelafaxState:
        config = llama_model.LlamaConfig()  # You might want to load this from somewhere
        dummy_state = LlamaModel.create_train_state(jax.random.PRNGKey(0), config)
        state_shapes = jax.eval_shape(lambda: dummy_state)
        state_shapes_partitioned = utils.match_partition_rules(
            llama_model.LlamaConfig.get_partition_rules(), state_shapes
        )
        shard_fns, _ = utils.make_shard_and_gather_fns(
            state_shapes_partitioned, state_shapes
        )
        loaded_state, restored_params = checkpointer.load_trainstate_checkpoint(
            path, state_shapes, shard_fns
        )
        if restored_params is not None:
            return FelafaxState.create(
                apply_fn=dummy_state.apply_fn,
                params=restored_params,
                tx=dummy_state.tx,
                config=loaded_state.config,
                model=dummy_state.model
            )
        return dummy_state


class Trainer:
    def __init__(self, model: FelafaxModule, config: Any):
        self.model = model
        self.config = config
        self.mesh = None
        self.state = None
        self.sharded_train_step = None

    def setup(self):
        utils.init_rng(self.config.seed)
        devices = jax.devices()
        device_count = len(devices)
        device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
        self.mesh = Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))

        state_shapes = self.model._get_state_shapes(self.config.max_length)
        state_shapes_partitioned = utils.match_partition_rules(
            llama_model.LlamaConfig.get_partition_rules(), state_shapes
        )

        self.shard_fns, self.gather_fns = utils.make_shard_and_gather_fns(
            state_shapes_partitioned, state_shapes
        )

        self.sharded_train_step = self._get_sharded_train_step(state_shapes_partitioned)
        self.sharded_create_trainstate = self._get_sharded_create_trainstate(state_shapes_partitioned)

    def _get_sharded_train_step(self, state_partitioned):
        return pjit(
            self.model.train_step,
            in_shardings=(state_partitioned, PS(), PS()),
            out_shardings=(state_partitioned, PS(), PS()),
            donate_argnums=(0, 1),
        )

    def _get_sharded_create_trainstate(self, state_partitioned):
        return pjit(
            create_trainstate_from_params,
            in_shardings=(state_partitioned.params,),
            out_shardings=state_partitioned,
            static_argnums=(1, 2),
        )

    def train_and_eval(self, train_dataloader, eval_dataloader, checkpointer):
        with self.mesh:
            if self.config.load_checkpoint:
                self.state = self.model.load_checkpoint(
                    self.config.checkpoint_path,
                    self.config.max_length,
                    checkpointer
                )
            else:
                # Initialize state if not loading from checkpoint
                rng = jax.random.PRNGKey(0)
                self.state = self.model._init_fn(rng, self.config.max_length)

            for epoch in range(self.config.num_epochs):
                print(f"Starting epoch {epoch} of training...")
                
                for step, train_batch in enumerate(train_dataloader):
                    train_batch = jax.device_put(train_batch, NamedSharding(self.mesh, PS()))
                    sharded_rng = utils.next_rng()

                    self.state, sharded_rng, metrics = self.sharded_train_step(
                        self.state, sharded_rng, train_batch
                    )

                    if step % self.config.print_every_n_steps == 0:
                        print(
                            f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                        )

                    if self.config.max_steps and step >= self.config.max_steps:
                        break

                # Evaluation
                eval_metrics = self._evaluate(eval_dataloader)
                print(f"Epoch {epoch}, Eval Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")

                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0:
                    self.model.save_checkpoint(self.state, f"checkpoint_epoch_{epoch}", checkpointer)

    def _evaluate(self, eval_dataloader):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for eval_batch in eval_dataloader:
            eval_batch = jax.device_put(eval_batch, NamedSharding(self.mesh, PS()))
            loss, accuracy = self.model.eval_step(self.state, eval_batch)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches
        }

    def export(self, export_path):
        # Implement model export logic here
        # This could involve converting the model to a specific format,
        # saving it in a deployable state, etc.
        pass
    
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
