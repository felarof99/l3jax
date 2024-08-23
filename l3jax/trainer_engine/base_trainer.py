import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from typing import Any, Dict, Tuple

class BaseTrainer:
    def __init__(self, model, optimizer, training_config):
        self.model = model
        self.optimizer = optimizer
        self.training_config = training_config

    def create_train_state(self, rng):
        """Initialize the train state."""
        raise NotImplementedError

    @staticmethod
    def loss_fn(logits, labels, mask=None):
        """Defines the loss function."""
        raise NotImplementedError

    def train_step(self, state, batch):
        """Defines a single training step."""
        raise NotImplementedError

    def save_checkpoint(self, state, path):
        """Saves a checkpoint of the model."""
        raise NotImplementedError

    def load_checkpoint(self, path):
        """Loads a checkpoint of the model."""
        raise NotImplementedError

    def train_and_eval(self, train_loader, eval_loader=None, num_epochs=1):
        """Interleaves training and evaluation."""
        rng = jax.random.PRNGKey(self.training_config.seed)
        state = self.create_train_state(rng)

        for epoch in range(num_epochs):
            rng, train_rng = jax.random.split(rng)
            state = self.train_epoch(state, train_loader)
            
            if eval_loader is not None:
                eval_metrics = self.evaluate_model(state, eval_loader)
                print(f"Epoch {epoch+1}/{num_epochs} - Eval metrics: {eval_metrics}")

            if (epoch + 1) % self.training_config.save_every == 0:
                self.save_checkpoint(state, f"checkpoint_epoch_{epoch+1}.ckpt")

        return state