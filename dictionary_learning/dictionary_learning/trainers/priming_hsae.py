import torch as t
from torch import nn
import torch.nn.functional as F

from .hierarchical_batch_top_k import HierarchicalBatchTopKTrainer, HierarchicalBatchTopKSAE

class ContextEncoder(nn.Module):
    """A simple encoder to create a context vector from the input activations."""
    def __init__(self, activation_dim: int, context_dim: int):
        super().__init__()
        self.layer = nn.Linear(activation_dim, context_dim)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Simple average pooling over the batch dimension can be one way,
        # but for now, we process each item in the batch independently.
        return F.relu(self.layer(x.detach()))

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer."""
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.layer = nn.Linear(context_dim, 2 * feature_dim) # 2 for gamma and beta

    def forward(self, features: t.Tensor, context: t.Tensor) -> t.Tensor:
        """
        Modulates features with the context vector.
        features: The activations from a dictionary layer.
        context: The context vector for the batch.
        """
        gamma_beta = self.layer(F.relu(context))
        gamma, beta = t.chunk(gamma_beta, 2, dim=-1)

        # Modulate the features
        return (gamma * features) + beta

class PrimingHSAE(HierarchicalBatchTopKSAE):
    """
    Priming Hierarchical SAE.
    This model uses a top-down signal (context vector) to modulate feature extraction
    at each level of the hierarchy, allowing for context-aware feature learning.
    """
    def __init__(self, context_dim, modulator_hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_dim = context_dim
        self.modulator_hidden_dim = modulator_hidden_dim

        # 1. Context Encoder
        self.context_encoder = ContextEncoder(self.activation_dim, self.context_dim)

        # 2. FiLM Layers for each level of the hierarchy
        self.film_layers = nn.ModuleList()
        # For level 0
        self.film_layers.append(FiLMLayer(self.context_dim, self.dict_size))
        # For subsequent levels
        for latent_size in self.lower_level_latent_sizes:
            self.film_layers.append(FiLMLayer(self.context_dim, latent_size))

    def forward(self, x: t.Tensor, y: t.Tensor = None):
        # 1. Generate context vector
        context_vector = self.context_encoder(x)

        all_latents = []
        current_latents = x

        # 2. Modulated bottom-up pass
        # Level 0
        l0_activations = self.encoders[0](current_latents)
        l0_modulated = self.film_layers[0](l0_activations, context_vector)
        l0_sparse_latents = self.get_sparse_latents(l0_modulated, self.k)
        all_latents.append(l0_sparse_latents)
        current_latents = l0_sparse_latents

        # Higher levels
        for i, k_lower in enumerate(self.lower_level_ks):
            latent_activations = self.encoders[i+1](current_latents)
            modulated_activations = self.film_layers[i+1](latent_activations, context_vector)
            sparse_latents = self.get_sparse_latents(modulated_activations, k_lower)
            all_latents.append(sparse_latents)
            current_latents = sparse_latents

        # 3. Decode from the final layer's latents
        recons = self.decode(current_latents)

        return recons, all_latents, None # Match parent signature

class PrimingHSAETrainer(HierarchicalBatchTopKTrainer):
    """
    Trainer for the PrimingHSAE. The training logic is identical to the standard
    Hierarchical Trainer, as the modulation is part of the model's forward pass.
    """
    def __init__(self, model: PrimingHSAE, **kwargs):
        super().__init__(model, **kwargs)