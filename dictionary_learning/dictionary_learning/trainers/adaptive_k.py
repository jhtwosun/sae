import torch as t
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from .trainer import SAETrainer
from ..dictionary import Dictionary
from .batch_top_k import BatchTopKSAE

# K-Predictor Network
class KPredictor(nn.Module):
    """
    A simple MLP that predicts an appropriate k value for a given input activation.
    """
    def __init__(self, activation_dim: int, hidden_dim: int, max_k: int):
        super().__init__()
        self.layer1 = nn.Linear(activation_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.max_k = max_k

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Forward pass to predict k.
        x: shape (batch_size, activation_dim)
        returns: shape (batch_size,)
        """
        # Detach to not propagate gradients from k prediction back to the main model
        x = x.detach()
        hidden = F.relu(self.layer1(x))
        # Use softplus to ensure output is positive, and add 1 to ensure k is at least 1.
        # Clamp the output to a maximum k to prevent runaway values.
        predicted_k = t.clamp(F.softplus(self.layer2(hidden).squeeze(-1)) + 1, min=1.0, max=self.max_k)
        return predicted_k

# The new SAE model
class AdaptiveKSAE(BatchTopKSAE):
    """
    This SAE uses a K-Predictor to dynamically determine k for each input.
    It inherits from BatchTopKSAE to reuse its structure but overrides the forward pass.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.k_predictor = KPredictor(
            activation_dim=cfg.activation_dim,
            hidden_dim=cfg.k_predictor_hidden_dim,
            max_k=cfg.dict_size // 2 # Max k can be half the dictionary size
        )
        # We will not use the fixed k from the config
        self.k = None

    def forward(self, x: t.Tensor, y: t.Tensor = None):
        # Get the predicted k for each item in the batch
        # Round to nearest integer and clamp to be safe
        predicted_k = t.round(self.k_predictor(x)).long()

        # Encoder
        latent_activations = self.encode(x)

        # Find top-k activations for each input in the batch
        # This logic is adapted from BatchTopKSAE but uses the predicted_k tensor
        top_k_values, top_k_indices = t.topk(latent_activations, k=self.k_predictor.max_k, dim=-1)

        # Create a mask for each item in the batch based on its predicted k
        k_mask = t.arange(self.k_predictor.max_k, device=x.device)[None, :] < predicted_k[:, None]

        # Mask out the values that are not in the top predicted_k
        masked_top_k_values = top_k_values * k_mask

        # Create the sparse latent tensor
        sparse_latents = t.zeros_like(latent_activations)
        sparse_latents.scatter_(dim=-1, index=top_k_indices, src=masked_top_k_values)

        # Decoder
        recons = self.decode(sparse_latents)

        # Return all values needed for the trainer's loss function
        return recons, sparse_latents, latent_activations, predicted_k


# The new Trainer
class AdaptiveKTrainer(SAETrainer):
    """
    Trainer for the AdaptiveKSAE.
    Implements the dual loss function.
    """
    def __init__(self, model: AdaptiveKSAE, **kwargs):
        super().__init__(model, **kwargs)
        self.k_predictor_alpha = model.cfg.k_predictor_alpha

    def calculate_loss(
        self,
        recons: t.Tensor,
        x: t.Tensor,
        latents: t.Tensor,
        all_latents: t.Tensor,
        predicted_k: t.Tensor,
        step: int,
    ) -> tuple[t.Tensor, dict]:

        # 1. Reconstruction Loss (MSE)
        recons_loss = F.mse_loss(recons, x)

        # 2. Sparsity Cost for the K-Predictor
        # We want to minimize the average k value predicted
        sparsity_cost = self.k_predictor_alpha * predicted_k.float().mean()

        # Total Loss
        total_loss = recons_loss + sparsity_cost

        # Metrics for logging
        with t.no_grad():
            # L0 is the average number of non-zero elements in the sparse latents
            l0 = (latents > 0).float().sum(dim=-1).mean()
            # Also log the average predicted k
            avg_predicted_k = predicted_k.float().mean()

        metrics = {
            "loss": total_loss.item(),
            "recons_loss": recons_loss.item(),
            "sparsity_cost": sparsity_cost.item(),
            "l0": l0.item(),
            "avg_predicted_k": avg_predicted_k.item(),
        }

        return total_loss, metrics

    def training_step(self, x: t.Tensor, step: int):
        recons, latents, all_latents, predicted_k = self.model(x)
        loss, metrics = self.calculate_loss(recons, x, latents, all_latents, predicted_k, step)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        if self.verbose:
            print(f"Step {step}: loss={loss.item():.4f}, l0={metrics['l0']:.2f}, avg_k={metrics['avg_predicted_k']:.2f}")

        return metrics