import torch as t
from torch import nn
import torch.nn.functional as F

from .trainer import SAETrainer
from .standard import StandardTrainer
from ..dictionary import AutoEncoder

class Router(nn.Module):
    """Outputs a probability distribution over experts for a given input."""
    def __init__(self, activation_dim: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(activation_dim, num_experts)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return F.softmax(self.layer(x.detach()), dim=-1)

class Expert(nn.Module):
    """A single expert, which is a standard SAE."""
    def __init__(self, activation_dim: int, expert_dict_size: int):
        super().__init__()
        self.encoder = nn.Linear(activation_dim, expert_dict_size)
        self.decoder = nn.Linear(expert_dict_size, activation_dim, bias=False)
        # Initialize decoder weights to be orthogonal
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: t.Tensor):
        latents = F.relu(self.encoder(x))
        recons = self.decoder(latents)
        return recons, latents

class GRSAE(nn.Module):
    """
    Gated Router SAE.
    This model uses a router to select a weighted combination of experts
    to represent the input, allowing for specialized, conditional feature learning.
    """
    def __init__(self, cfg):
        super().__init__()
        self.activation_dim = cfg.activation_dim
        self.num_experts = cfg.num_experts
        self.expert_dict_size = cfg.expert_dict_size

        self.router = Router(self.activation_dim, self.num_experts)
        self.experts = nn.ModuleList(
            [Expert(self.activation_dim, self.expert_dict_size) for _ in range(self.num_experts)]
        )

    def forward(self, x: t.Tensor, y: t.Tensor = None):
        # 1. Get routing weights
        routing_weights = self.router(x) # (batch_size, num_experts)

        # 2. Get outputs from all experts
        all_recons = []
        all_latents = []
        for expert in self.experts:
            recons, latents = expert(x)
            all_recons.append(recons)
            all_latents.append(latents)

        # 3. Combine expert outputs using routing weights
        # Stack experts outputs along a new dimension
        # recons shape: (batch_size, num_experts, activation_dim)
        # latents shape: (batch_size, num_experts, expert_dict_size)
        recons_stack = t.stack(all_recons, dim=1)
        latents_stack = t.stack(all_latents, dim=1)

        # Weigh the outputs by the router
        # routing_weights need to be reshaped to (batch_size, num_experts, 1) for broadcasting
        weighted_recons = t.sum(routing_weights.unsqueeze(-1) * recons_stack, dim=1)
        weighted_latents = t.sum(routing_weights.unsqueeze(-1) * latents_stack, dim=1)

        return weighted_recons, weighted_latents, routing_weights

class GRSAETrainer(StandardTrainer):
    """
    Trainer for the GRSAE.
    This trainer adds a load-balancing auxiliary loss to the standard reconstruction loss
    to ensure that all experts are utilized relatively evenly.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.routing_loss_alpha = self.model.cfg.routing_loss_alpha
        
            # super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.
    
    def calculate_loss(
        self,
        recons: t.Tensor,
        x: t.Tensor,
        latents: t.Tensor,
        routing_weights: t.Tensor,
        step: int,
    ) -> tuple[t.Tensor, dict]:
        
        # 1. Reconstruction Loss (MSE)
        recons_loss = F.mse_loss(recons, x)

        # 2. Load Balancing Loss for Router
        # This loss encourages the router to use all experts relatively equally.
        # It's based on the product of the average usage of each expert.
        avg_routing_per_expert = routing_weights.mean(dim=0) # shape: (num_experts,)
        load_balancing_loss = self.num_experts * t.sum(avg_routing_per_expert * avg_routing_per_expert)
        
        # 3. L1 Sparsity Loss on latents (from parent class)
        l1_loss = self.l1_penalty * t.abs(latents).sum(dim=-1).mean()

        total_loss = recons_loss + l1_loss + (self.routing_loss_alpha * load_balancing_loss)

        with t.no_grad():
            l0 = (latents > 0).float().sum(dim=-1).mean()

        metrics = {
            "loss": total_loss.item(),
            "recons_loss": recons_loss.item(),
            "l1_loss": l1_loss.item(),
            "load_balancing_loss": load_balancing_loss.item(),
            "l0": l0.item(),
        }

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
            
        return total_loss, metrics

    def training_step(self, x: t.Tensor, step: int):
        recons, latents, routing_weights = self.model(x)
        loss, metrics = self.calculate_loss(recons, x, latents, routing_weights, step)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        if self.verbose:
            print(f"Step {step}: loss={loss.item():.4f}, l0={metrics['l0']:.2f}")

        return metrics


    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }


    