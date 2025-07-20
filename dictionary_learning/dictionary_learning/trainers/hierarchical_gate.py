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

class HierarchicalSAE_Gated(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, lower_level_latent_sizes: List[int], lower_level_ks: List[int]):
        super().__init__()
        
        prod_lower_sizes = math.prod(lower_level_latent_sizes) if lower_level_latent_sizes else 1
        prod_lower_ks = math.prod(lower_level_ks) if lower_level_ks else 1

        assert dict_size > 0 and dict_size % prod_lower_sizes == 0
        assert k > 0 and k % prod_lower_ks == 0

        s0, k0 = dict_size // prod_lower_sizes, k // prod_lower_ks
        self.latent_sizes_list, self.ks_list = [s0] + lower_level_latent_sizes, [k0] + lower_level_ks
        self.n_levels, self.activation_dim, self.total_dict_size = len(self.latent_sizes_list), activation_dim, dict_size

        self.register_buffer("dict_size", t.tensor(dict_size, dtype=t.int64))
        self.register_buffer("k", t.tensor(k, dtype=t.int64))
        self.register_buffer("lower_level_latent_sizes", t.tensor(lower_level_latent_sizes, dtype=t.int64))
        self.register_buffer("lower_level_ks", t.tensor(lower_level_ks, dtype=t.int64))
        
        self.decoder = nn.Linear(self.total_dict_size, self.activation_dim, bias=False)
        self.b_dec = nn.Parameter(t.zeros(self.activation_dim))
        
        self.encoders = nn.ModuleList([
            nn.Linear(self.activation_dim, int(np.prod(self.latent_sizes_list[:i+1])), bias=True)
            for i in range(self.n_levels)
        ])
        
        ## 변경/추가된 부분: 각 계층 결합을 위한 게이트 모듈
        # n_levels-1개의 게이트가 필요합니다 (level 0과 1 결합, level 1과 2 결합 등)
        self.gates = nn.ModuleList([
            nn.Linear(int(np.prod(self.latent_sizes_list[:i+1])), int(np.prod(self.latent_sizes_list[:i+1])))
            for i in range(self.n_levels - 1)
        ])
        
        self.thresholds = nn.ParameterList([
            nn.Parameter(t.full((int(np.prod(self.latent_sizes_list[:i])) if i > 0 else 1,), -1.0), requires_grad=False)
            for i in range(self.n_levels)
        ])

        set_decoder_norm_to_unit_norm(self.decoder.weight, self.activation_dim, self.total_dict_size)
        with t.no_grad():
            self.encoders[-1].weight.data = self.decoder.weight.data.T.clone()
            for i, encoder in enumerate(self.encoders):
                if i < len(self.encoders) - 1:  # 마지막 이전 것들
                    encoder.weight.data.normal_(mean=0.0, std=0.001)
                    encoder.bias.data.fill_(1.0)
                else:
                    # 마지막 인코더의 bias는 원한다면 별도로 지정하거나 그대로 둘 수 있음
                    encoder.bias.data.zero_()
 

    def scale_biases(self, scale: float):
        with t.no_grad():
            self.b_dec.data *= scale
            for encoder in self.encoders:
                encoder.bias.data *= scale
            for threshold_tensor in self.thresholds:
                positive_mask = threshold_tensor.data >= 0
                threshold_tensor.data[positive_mask] *= scale


    def _batch_topk(self, acts: t.Tensor, k: int) -> t.Tensor:
        # acts: (B, N)
        if k == 0 or acts.numel() == 0:
            return t.zeros_like(acts)
        values, indices = t.topk(acts, k, dim=-1)
        mask = t.zeros_like(acts)
        mask.scatter_(dim=-1, index=indices, src=t.ones_like(values))
        return acts * mask

    def _vectorized_group_topk(self, acts: t.Tensor, k: int) -> t.Tensor:
        # acts: (B, G, S)
        B, G, S = acts.shape
        if k == 0 or acts.numel() == 0:
            return t.zeros_like(acts)
        acts_flat = acts.view(B * G, S)
        values, indices = t.topk(acts_flat, k, dim=-1)
        mask = t.zeros_like(acts_flat)
        mask.scatter_(dim=-1, index=indices, src=t.ones_like(values))
        sparse_flat = acts_flat * mask
        return sparse_flat.view(B, G, S)
        
    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = False):
        batch_size = x.shape[0]
        x_centered = x - self.b_dec
        level_latents_sparse = []
        for i in range(self.n_levels):
            post_relu_acts = F.relu(self.encoders[i](x_centered))
            if i == self.n_levels - 1 and return_active:
                post_relu_acts_last_layer = post_relu_acts.clone()
            
            if i == 0:
                sparse_acts = self._batch_topk(post_relu_acts, self.ks_list[i])
            else:
                num_groups, group_size = int(np.prod(self.latent_sizes_list[:i])), self.latent_sizes_list[i]
                grouped_acts = post_relu_acts.view(batch_size, num_groups, group_size)
                sparse_acts = self._vectorized_group_topk(grouped_acts, self.ks_list[i])
            level_latents_sparse.append(sparse_acts)


        final_latent = level_latents_sparse[-1].view(batch_size, *self.latent_sizes_list)

        for i in range(self.n_levels - 2, -1, -1):
            # 게이트의 입력으로 사용할 하위 레벨의 희소 잠재 벡터
            latent_to_gate = level_latents_sparse[i]
            
            # 게이트 통과 후 시그모이드 함수로 0~1 사이의 값으로 변환
            # 이 값이 상위 레벨의 특성을 얼마나 통과시킬지 결정
            gate_signal = t.sigmoid(self.gates[i](latent_to_gate))

            # 브로드캐스팅을 위해 게이트 신호의 형태를 맞춰줌
            target_shape = [batch_size] + list(self.latent_sizes_list[:i+1]) + [1] * (len(self.latent_sizes_list) - (i+1))
            gate_signal_reshaped = gate_signal.view(*target_shape)

            # 단순 곱셈 대신 게이트를 통과한 신호를 곱해줌
            final_latent = final_latent * gate_signal_reshaped
            
        final_latent_flat = final_latent.reshape(batch_size, -1)

        if not return_active:
            return final_latent_flat
        else:
            active_indices_F = (final_latent_flat.sum(0) > 0)
            return final_latent_flat, active_indices_F, post_relu_acts_last_layer, level_latents_sparse

    def decode(self, x_encoded: t.Tensor) -> t.Tensor:
        return self.decoder(x_encoded) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, use_threshold: bool = False):
        encoded_acts = self.encode(x, use_threshold=use_threshold)
        x_hat = self.decode(encoded_acts)
        return (x_hat, encoded_acts) if output_features else x_hat

    @classmethod
    def from_pretrained(cls, path, k, lower_level_latent_sizes, lower_level_ks, device: Optional[str] = None):
        state_dict = t.load(path, map_location=device or 'cpu')
        model = cls(
            activation_dim=state_dict['b_dec'].shape[0], dict_size=state_dict['dict_size'].item(),
            k=k, lower_level_latent_sizes=lower_level_latent_sizes,
            lower_level_ks=lower_level_ks
        )
        model.load_state_dict(state_dict)
        if device: model.to(device)
        return model

class HierarchicalSAEGatedTrainer:
    def __init__(
        self, steps: int, activation_dim: int, dict_size: int, k: int, 
        lower_level_latent_sizes: List[int], lower_level_ks: List[int],
        layer: int, lm_name: str, dict_class: type = HierarchicalSAE_Gated,
        lr: Optional[float] = None, auxk_alpha: float = 1 / 32, warmup_steps: int = 1000,
        decay_start: Optional[int] = None, threshold_beta: float = 0.999,
        threshold_start_step: int = 1000, seed: Optional[int] = None,
        device: Optional[str] = None, wandb_name: str = "HierarchicalSAE_Gated",
        submodule_name: Optional[str] = None,
    ):
        if seed is not None: t.manual_seed(seed); t.cuda.manual_seed_all(seed)
        self.steps, self.activation_dim, self.dict_size, self.k = steps, activation_dim, dict_size, k
        self.lower_level_latent_sizes, self.lower_level_ks = lower_level_latent_sizes, lower_level_ks
        self.layer, self.lm_name, self.submodule_name = layer, lm_name, submodule_name
        self.wandb_name, self.auxk_alpha = wandb_name, auxk_alpha
        self.warmup_steps, self.decay_start = warmup_steps, decay_start
        self.threshold_beta, self.threshold_start_step, self.seed = threshold_beta, threshold_start_step, seed
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]

        self.ae = dict_class(
            activation_dim=self.activation_dim, dict_size=self.dict_size, k=self.k,
            lower_level_latent_sizes=self.lower_level_latent_sizes,
            lower_level_ks=self.lower_level_ks
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
            prev = guess.clone()
            weights = 1.0 / t.norm(points - guess, dim=1, p=2)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol: break
        return guess

    def get_logging_parameters(self):
        log_dict = {}
        for key in self.logging_parameters:
            log_dict[key] = getattr(self, key)
        return log_dict

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

    def update_threshold(self, level_latents_sparse: List[t.Tensor]):
        for i, level_latent in enumerate(level_latents_sparse):
            target_thresholds = self.ae.thresholds[i]
            device_type = "cuda" if level_latent.is_cuda else "cpu"

            with t.autocast(device_type=device_type, enabled=False), t.no_grad():
                if level_latent.dim() == 2:
                    active_acts = level_latent[level_latent > 0]
                    if active_acts.numel() == 0:
                        min_activation = t.tensor(0.0, dtype=t.float32, device=level_latent.device)
                    else:
                        min_activation = active_acts.min().detach().to(dtype=t.float32)

                    if target_thresholds[0] < 0:
                        new_threshold = min_activation
                    else:
                        new_threshold = (self.threshold_beta * target_thresholds[0].to(t.float32)) + \
                                        ((1 - self.threshold_beta) * min_activation)

                    target_thresholds[0] = new_threshold.to(dtype=target_thresholds.dtype)


                elif level_latent.dim() == 3:
                    B, G, D = level_latent.shape
                    active_acts = t.where(level_latent > 0,
                                        level_latent,
                                        t.full_like(level_latent, float('inf')))
                    min_per_group = active_acts.reshape(B * G, D).min(dim=1).values
                    min_per_group = min_per_group.view(B, G).min(dim=0).values

                    has_min_val = min_per_group != float('inf')
                    if has_min_val.any():
                        curr = target_thresholds[has_min_val].to(t.float32)
                        mins = min_per_group[has_min_val].detach().to(dtype=t.float32)
                        new_vals = curr.clone()

                        init_mask = curr < 0
                        ema_mask = ~init_mask

                        if init_mask.any():
                            new_vals[init_mask] = mins[init_mask]
                        if ema_mask.any():
                            new_vals[ema_mask] = (self.threshold_beta * curr[ema_mask]) + \
                                                ((1 - self.threshold_beta) * mins[ema_mask])

                        target_thresholds[has_min_val] = new_vals.to(dtype=target_thresholds.dtype)

    def loss(self, x: t.Tensor, step: Optional[int] = None, logging: bool = False):
        f, active_indices_F, post_relu_acts_BF, level_latents_sparse = self.ae.encode(x, return_active=True)
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(level_latents_sparse)
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



