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
from flax.serialization import from_bytes, from_state_dict, to_bytes, to_state_dict
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

# Local imports


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
    felafax_module: "FxModule"


class FxModule(ABC):
    @staticmethod
    @abstractmethod
    def setup(training_config: Any) -> FxTrainState:
        """Initializes the training state with the given configuration."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def create_train_state(rng: Any, training_config: Any) -> FxTrainState:
        """Creates a new training state with initialized model parameters."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def train_step(
        state: FxTrainState, rng: Any, batch: Any
    ) -> Tuple[FxTrainState, Any, dict]:
        """Performs a single training step and returns updated state and metrics."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def eval_step(state: FxTrainState, batch: Any) -> Tuple[float, float]:
        """Evaluates the model on a single batch and returns loss and accuracy."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def save_checkpoint(state: FxTrainState, path: str, checkpointer: Any) -> None:
        """Saves the current training state to a checkpoint file."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_checkpoint(path: str, seq_length: int, checkpointer: Any) -> FxTrainState:
        """Loads a training state from a checkpoint file."""
        raise NotImplementedError()


########################################################################
# LlamaModel class implementation
# now create llama model class that inherits from FelafaxModule
########################################################################
class FxLlamaModel(FxModule):
    @staticmethod
    def setup(training_config: FxTrainingConfig) -> FxTrainState:
        return FxLlamaModel.create_train_state(jax.random.PRNGKey(0), training_config)

    @staticmethod
    def create_train_state(rng: Any, training_config: Any) -> FxTrainState:
        llama_config = llama_model.LlamaConfig()
        model = llama_model.CausalLlamaModule(
            llama_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        params = FxLlamaModel._init_params(rng, training_config.seq_length, model)
        tx = optax.sgd(training_config.learning_rate)
        return FxTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            config=training_config,
            model=model,
        )

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
    def train_step(
        state: FxTrainState, rng: Any, batch: Any
    ) -> Tuple[FxTrainState, Any, dict]:
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
    def eval_step(state: FxTrainState, batch: Any) -> Tuple[float, float]:
        logits = state.apply_fn(
            state.params,
            batch["input_tokens"],
            deterministic=True,
        ).logits
        return utils.cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )

    @staticmethod
    def compute_loss(state: FxTrainState, batch: Any) -> float:
        return FxLlamaModel.eval_step(state, batch)[0]  # Return only the loss

    @staticmethod
    def save_checkpoint(state: FxTrainState, path: str, checkpointer: Any) -> None:
        checkpointer.save_trainstate_checkpoint(state, path)

    @staticmethod
    def load_checkpoint(path: str, seq_length: int, checkpointer: Any) -> FxTrainState:
        config = llama_model.LlamaConfig()  # You might want to load this from somewhere
        dummy_state = FxLlamaModel.create_train_state(jax.random.PRNGKey(0), config)
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
            return FxTrainState.create(
                apply_fn=dummy_state.apply_fn,
                params=restored_params,
                tx=dummy_state.tx,
                config=loaded_state.config,
                model=dummy_state.model,
            )
        return dummy_state


import functools
from typing import Any

########################################################################
# Trainer class implementation
# Felafax trainer class that takes in a model and config and sets up the training loop
########################################################################
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding

from . import checkpoint_lib, llama_model, utils


class Trainer:
    def __init__(self, model: FxModule, config: Any):
        self.model = model
        self.config = config
        self.mesh = None
        self.state = None
        self.sharded_train_step = None
        self.sharded_create_train_state = None
        self.shard_fns = None
        self.gather_fns = None

    def setup(self):
        utils.init_rng(self.config.seed)
        devices = jax.devices()
        device_count = len(devices)
        device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
        self.mesh = Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))

        dummy_state = self.model.create_train_state(jax.random.PRNGKey(0), self.config)
        state_shapes = jax.eval_shape(lambda: dummy_state)
        state_shapes_partitioned = utils.match_partition_rules(
            llama_model.LlamaConfig.get_partition_rules(), state_shapes
        )

        self.shard_fns, self.gather_fns = utils.make_shard_and_gather_fns(
            state_shapes_partitioned, state_shapes
        )

        self.sharded_train_step = self._get_sharded_train_step(state_shapes_partitioned)
        self.sharded_create_train_state = self._get_sharded_create_train_state(
            state_shapes_partitioned
        )

    def _get_sharded_train_step(self, state_partitioned):
        return pjit(
            self.model.train_step,
            in_shardings=(
                state_partitioned,
                jax.sharding.PartitionSpec(),
                jax.sharding.PartitionSpec(),
            ),
            out_shardings=(
                state_partitioned,
                jax.sharding.PartitionSpec(),
                jax.sharding.PartitionSpec(),
            ),
            donate_argnums=(0, 1),
        )

    def _get_sharded_create_train_state(self, state_partitioned):
        return pjit(
            self.model.create_train_state,
            in_shardings=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
            out_shardings=state_partitioned,
            static_argnums=(1,),
        )

    def train_and_eval(self, train_dataloader, eval_dataloader, checkpointer):
        with self.mesh:
            if self.config.load_checkpoint:
                self.state = self.model.load_checkpoint(
                    self.config.checkpoint_path, self.config.max_length, checkpointer
                )
            else:
                # Initialize state if not loading from checkpoint
                self.state = self.sharded_create_train_state(
                    jax.random.PRNGKey(0), self.config
                )

            for epoch in range(self.config.num_epochs):
                print(f"Starting epoch {epoch} of training...")

                for step, train_batch in enumerate(train_dataloader):
                    train_batch = jax.device_put(
                        train_batch,
                        NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
                    )
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
                print(
                    f"Epoch {epoch}, Eval Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}"
                )

                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0:
                    self.model.save_checkpoint(
                        self.state, f"checkpoint_epoch_{epoch}", checkpointer
                    )

    def _evaluate(self, eval_dataloader):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for eval_batch in eval_dataloader:
            eval_batch = jax.device_put(
                eval_batch, NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            )
            loss, accuracy = self.model.eval_step(self.state, eval_batch)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
        }

    def export(self, export_path):
        # Implement model export logic here
        checkpointer = checkpoint_lib.Checkpointer(
            checkpoint_lib.Checkpointer.get_default_config(),
            checkpoint_dir=os.path.dirname(export_path),
            enable_checkpointer=jax.process_index() == 0,
        )
        checkpointer.save_train_state_to_file(
            train_state=self.state,
            gather_fns=self.gather_fns,
            path=export_path,
        )


# Usage example (to be placed in llama3.1_train.py or similar):
def main():
    training_cfg = FxTrainingConfig()

    # Initialize model
    llama_config = llama_model.LlamaConfig.get_standard_llama_config("llama3_8b")
    llama_config = llama_model.LlamaConfig.finalize_config(llama_config)
    model = FxLlamaModel(llama_config, dtype=jnp.float32, param_dtype=jnp.float32)

    # Initialize trainer
    trainer = Trainer(model, training_cfg)
    trainer.setup()

    # Prepare dataset
    train_dataloader, val_dataloader = get_dataset(
        tokenizer=tokenizer,
        max_length=training_cfg.max_length,
        max_examples=training_cfg.dataset_size_limit,
    )

    # Initialize checkpointer
    checkpointer = checkpoint_lib.Checkpointer(
        checkpoint_lib.Checkpointer.get_default_config(),
        checkpoint_dir=os.path.dirname(model_path),
        enable_checkpointer=jax.process_index() == 0,
    )

    # Train the model
    trainer.train_and_eval(train_dataloader, val_dataloader, checkpointer)

    # Export the model
    trainer.export(os.path.join(model_path, "trained_llama.flax"))


if __name__ == "__main__":
    main()


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
