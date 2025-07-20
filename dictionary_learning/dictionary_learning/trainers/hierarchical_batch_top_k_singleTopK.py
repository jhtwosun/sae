import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np
from collections import namedtuple
from typing import Optional, List, Tuple, NamedTuple

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from ..trainers.batch_top_k import BatchTopKSAE


class HierarchicalBatchTopKSAE_singleTopK(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, lower_level_latent_sizes: List[int]):
        super().__init__()
        
        prod_lower_sizes = math.prod(lower_level_latent_sizes) if lower_level_latent_sizes else 1
        assert dict_size > 0 and dict_size % prod_lower_sizes == 0 and k > 0
        s0 = dict_size // prod_lower_sizes
        
        self.latent_sizes_list = [s0] + lower_level_latent_sizes
        self.n_levels, self.activation_dim, self.total_dict_size = len(self.latent_sizes_list), activation_dim, dict_size

        self.register_buffer("dict_size", t.tensor(dict_size, dtype=t.int64))
        self.register_buffer("k", t.tensor(k, dtype=t.int64))
        self.register_buffer("lower_level_latent_sizes", t.tensor(lower_level_latent_sizes, dtype=t.int64))
        
        self.decoder = nn.Linear(self.total_dict_size, self.activation_dim, bias=False)
        self.b_dec = nn.Parameter(t.zeros(self.activation_dim))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))
        
        self.encoders = nn.ModuleList([
            nn.Linear(self.activation_dim, int(np.prod(self.latent_sizes_list[:i+1])), bias=True)
            for i in range(self.n_levels)
        ])
        
        set_decoder_norm_to_unit_norm(self.decoder.weight, self.activation_dim, self.total_dict_size)
        with t.no_grad():
            self.encoders[-1].weight.data = self.decoder.weight.data.T.clone()
            for encoder in self.encoders: encoder.bias.data.zero_()

    def scale_biases(self, scale: float):
        with t.no_grad():
            self.b_dec.data *= scale
            for encoder in self.encoders: encoder.bias.data *= scale
            if self.threshold.item() >= 0:
                self.threshold *= scale

    def _batch_topk(self, acts: t.Tensor, k: int) -> t.Tensor:
        batch_size = acts.size(0); k_for_batch = k * batch_size
        if k_for_batch == 0 or acts.numel() == 0: return t.zeros_like(acts)
        k_for_batch = min(k_for_batch, acts.numel()); flattened_acts = acts.flatten()
        top_k_values, top_k_indices = t.topk(flattened_acts, k_for_batch, sorted=False)
        sparse_acts_flat = t.zeros_like(flattened_acts)
        sparse_acts_flat.scatter_(-1, top_k_indices, top_k_values)
        return sparse_acts_flat.reshape(acts.shape)

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = False):
        batch_size = x.shape[0]
        x_centered = x - self.b_dec
        
        # 1. 모든 레벨에서 ReLU를 통과한 밀집(dense) 활성화를 계산
        dense_level_latents = [F.relu(encoder(x_centered)) for encoder in self.encoders]

        # 2. 모든 밀집 활성화들을 계층적으로 곱하여 최종 밀집 특성 생성
        final_dense_latent = dense_level_latents[-1].view(batch_size, *self.latent_sizes_list)
        for i in range(self.n_levels - 1):
            latent_to_multiply = dense_level_latents[i]
            broadcast_shape = [batch_size] + self.latent_sizes_list[:i+1]
            num_singleton_dims = self.n_levels - len(broadcast_shape) + 1
            broadcast_shape.extend([1] * num_singleton_dims)
            final_dense_latent = final_dense_latent * latent_to_multiply.view(broadcast_shape)
        
        final_dense_flat = final_dense_latent.view(batch_size, -1)

        # 3. 최종 밀집 특성에 Batch Top-K를 적용하여 희소 코드 생성
        if use_threshold:
            final_sparse_flat = final_dense_flat * (final_dense_flat > self.threshold)
        else:
            k_for_batch = self.k.item() * batch_size
            if k_for_batch == 0 or final_dense_flat.numel() == 0:
                 final_sparse_flat = t.zeros_like(final_dense_flat)
            else:
                k_for_batch = min(k_for_batch, final_dense_flat.numel())
                
                flattened_acts = final_dense_flat.flatten()
                top_k_values, top_k_indices = t.topk(flattened_acts, k_for_batch, sorted=False)
                
                sparse_acts_flat = t.zeros_like(flattened_acts)
                sparse_acts_flat.scatter_(-1, top_k_indices, top_k_values)
                
                final_sparse_flat = sparse_acts_flat.reshape(final_dense_flat.shape)

        if not return_active:
            return final_sparse_flat
        else:
            active_indices_F = (final_sparse_flat.sum(0) > 0)
            # 보조 손실 계산을 위해 Top-K 적용 전의 최종 밀집 특성을 반환
            return final_sparse_flat, active_indices_F, final_dense_flat

    def decode(self, x_encoded: t.Tensor) -> t.Tensor:
        return self.decoder(x_encoded) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, use_threshold: bool = False):
        encoded_acts = self.encode(x, use_threshold=use_threshold)
        x_hat = self.decode(encoded_acts)
        return (x_hat, encoded_acts) if output_features else x_hat

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        state_dict = t.load(path, map_location=device or 'cpu')
        model = cls(
            activation_dim=state_dict['b_dec'].shape[0], dict_size=state_dict['dict_size'].item(),
            k=state_dict['k'].item(), lower_level_latent_sizes=state_dict['lower_level_latent_sizes'].tolist()
        )
        model.load_state_dict(state_dict)
        if device: model.to(device)
        return model

# ----------------- 트레이너 클래스 -----------------
class HierarchicalBatchTopKSAE_singleTopKTrainer:
    def __init__(
        self, steps: int, activation_dim: int, dict_size: int, k: int, 
        lower_level_latent_sizes: List[int], lower_level_ks: List[int],
        layer: int, lm_name: str, dict_class: type = HierarchicalBatchTopKSAE_singleTopK,
        lr: Optional[float] = None, auxk_alpha: float = 1 / 32, warmup_steps: int = 1000,
        decay_start: Optional[int] = None, threshold_beta: float = 0.999,
        threshold_start_step: int = 1000, seed: Optional[int] = None,
        device: Optional[str] = None, wandb_name: str = "HierarchicalBatchTopKSAE_singleTopK",
        submodule_name: Optional[str] = None, **kwargs, # lower_level_ks를 무시하기 위해 kwargs 추가
    ):
        if seed is not None: t.manual_seed(seed); t.cuda.manual_seed_all(seed)
        self.steps, self.activation_dim, self.dict_size, self.k = steps, activation_dim, dict_size, k
        self.lower_level_latent_sizes, self.lower_level_ks = lower_level_latent_sizes, lower_level_ks
        self.layer, self.lm_name, self.submodule_name = layer, lm_name, submodule_name
        self.wandb_name, self.auxk_alpha = wandb_name, auxk_alpha
        self.warmup_steps, self.decay_start, self.threshold_beta = warmup_steps, decay_start, threshold_beta
        self.threshold_start_step, self.seed = threshold_start_step, seed
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.ae = dict_class(
            activation_dim=self.activation_dim, dict_size=self.dict_size, k=self.k,
            lower_level_latent_sizes=self.lower_level_latent_sizes
        )
        self.device = device or ("cuda" if t.cuda.is_available() else "cpu")
        self.ae.to(self.device)
        self.lr = lr if lr is not None else (2e-4 / (self.ae.total_dict_size / (2**14))**0.5)
        self.dead_feature_threshold, self.top_k_aux = 10_000_000, activation_dim // 2
        self.num_tokens_since_fired = t.zeros(self.ae.total_dict_size, dtype=t.long, device=self.device)
        self.effective_l0, self.dead_features, self.pre_norm_auxk_loss = -1, -1, -1
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, get_lr_schedule(steps, warmup_steps, decay_start))

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        for _ in range(max_iter):
            prev = guess.clone(); weights = 1.0 / t.norm(points - guess, dim=1, p=2)
            weights /= weights.sum(); guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol: break
        return guess

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features_mask = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features_mask.sum())
        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)
            auxk_latents = t.where(dead_features_mask[None], post_relu_acts_BF, -t.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts_BF = t.zeros_like(post_relu_acts_BF).scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            self.pre_norm_auxk_loss = l2_loss_aux.item()
            residual_mu = residual_BD.mean(dim=0, keepdim=True)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            return (l2_loss_aux / (loss_denom + 1e-6)).nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1.0
            return t.tensor(0.0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        with t.no_grad():
            active_acts = f[f > 0]
            if active_acts.numel() == 0:
                min_activation = 0.0
            else:
                min_activation = active_acts.min().item()

            if self.ae.threshold.item() < 0:
                self.ae.threshold.data = t.tensor(min_activation, dtype=self.ae.threshold.dtype)
            else:
                new_threshold = self.threshold_beta * self.ae.threshold.item() + (1 - self.threshold_beta) * min_activation
                self.ae.threshold.data = t.tensor(new_threshold, dtype=self.ae.threshold.dtype)

    def loss(self, x: t.Tensor, step: Optional[int] = None, logging: bool = False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(x, return_active=True)
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)
        x_hat = self.ae.decode(f)
        e = x - x_hat
        self.effective_l0 = (f > 0).float().sum(dim=-1).mean().item()
        num_tokens_in_step = x.size(0)
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[active_indices_F] = 0
        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss
        if not logging: return loss
        else:
            LossLog = namedtuple("LossLog", ["x", "x_hat", "f", "losses"])
            return LossLog(x, x_hat, f, {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()})
    
    def get_logging_parameters(self):
        log_dict = {}
        for key in self.logging_parameters: log_dict[key] = getattr(self, key)
        return log_dict

    def update(self, step: int, x: t.Tensor):
        if step == 0:
            median = self.geometric_median(x).to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight, self.ae.decoder.weight.grad, self.ae.activation_dim, self.ae.total_dict_size
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        set_decoder_norm_to_unit_norm(self.ae.decoder.weight, self.ae.activation_dim, self.ae.total_dict_size)
        
    @property
    def config(self):
        return {
            "trainer_class": self.__class__.__name__, "dict_class": self.ae.__class__.__name__,
            "activation_dim": self.activation_dim, "dict_size": self.dict_size, "k": self.k,
            "lower_level_latent_sizes": self.lower_level_latent_sizes, "lower_level_ks": self.lower_level_ks,
            "lr": self.lr, "steps": self.steps, "auxk_alpha": self.auxk_alpha, "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start, "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step, "seed": self.seed, "device": self.device,
            "layer": self.layer, "lm_name": self.lm_name, "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }