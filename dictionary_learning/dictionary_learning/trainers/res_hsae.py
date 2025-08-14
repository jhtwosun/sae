import torch as t
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from .hierarchical_batch_top_k import HierarchicalBatchTopKTrainer, HierarchicalBatchTopKSAE

class ResHSAE(HierarchicalBatchTopKSAE):
    """
    Residual Hierarchical SAE.
    This model enhances the Hierarchical SAE by adding skip connections from the original
    input to each subsequent hierarchical level. This helps mitigate information loss
    and can improve training dynamics.
    """
    def __init__(self, *args, **kwargs):
        # We call the parent's __init__ but will override the encoder creation
        super().__init__(*args, **kwargs)

        self.activation_dim = kwargs["activation_dim"]
        self._dict_size = kwargs["dict_size"]
        self._lower_level_latent_sizes = kwargs["lower_level_latent_sizes"]

        # Overwrite the encoders with correct input dimensions
        self.encoders = nn.ModuleList()
        current_input_dim = self.activation_dim

        # Level 0 encoder
        self.encoders.append(
            nn.Linear(current_input_dim, self._dict_size)
        )

        # Higher-level encoders
        prev_level_output_dim = self._dict_size
        for i, latent_size in enumerate(self._lower_level_latent_sizes):
            # Input to this level is concatenation of previous level's output and original input
            current_input_dim = prev_level_output_dim + self.activation_dim
            self.encoders.append(
                nn.Linear(current_input_dim, latent_size)
            )
            prev_level_output_dim = latent_size

        # The final decoder's input dimension must match the last level's output dimension
        self.decoder = nn.Linear(prev_level_output_dim, self.activation_dim, bias=False)

    def forward(self, x: t.Tensor, y: t.Tensor = None):
        all_latents = []
        current_latents = x

        # Level 0
        l0_latent_activations = self.encoders[0](current_latents)
        l0_sparse_latents = self.get_sparse_latents(l0_latent_activations, self.k)
        all_latents.append(l0_sparse_latents)
        current_latents = l0_sparse_latents

        # Higher levels with skip connections
        for i, k_lower in enumerate(self.lower_level_ks):
            # Concatenate previous level's output with original input x
            residual_input = t.cat([current_latents, x], dim=-1)

            # Pass through the encoder for this level
            latent_activations = self.encoders[i+1](residual_input)
            sparse_latents = self.get_sparse_latents(latent_activations, k_lower)
            all_latents.append(sparse_latents)
            current_latents = sparse_latents

        # Decode from the final layer's latents
        recons = self.decode(current_latents)

        # The second returned value (all_latents) is not used in the standard trainer loss function
        # but could be used for analysis or auxiliary losses.
        return recons, all_latents, None # Returning None for all_latents to match parent signature

# The trainer can be the same as the hierarchical one, as the loss calculation is identical.
# We create a new class for clarity and to ensure it's paired with the correct model.
class ResHSAETrainer(HierarchicalBatchTopKTrainer):
    """
    Trainer for the ResHSAE. The training logic is identical to the standard
    Hierarchical Trainer, but this class ensures it is paired with the ResHSAE model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)