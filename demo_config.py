from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
from enum import Enum
import torch as t
import itertools
import math

from dictionary_learning.dictionary_learning.trainers.standard import (
    StandardTrainer,
    StandardTrainerAprilUpdate,
)
from dictionary_learning.dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)
from dictionary_learning.dictionary_learning.trainers.batch_top_k import (
    BatchTopKTrainer,
    BatchTopKSAE,
)
from dictionary_learning.dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)

from dictionary_learning.dictionary_learning.trainers.hierarchical_batch_top_k import (
    HierarchicalBatchTopKTrainer,
    HierarchicalBatchTopKSAE,
)

from dictionary_learning.dictionary_learning.trainers.hierarchical_batch_top_k_singleTopK import (
    HierarchicalBatchTopKSAE_singleTopKTrainer,
    HierarchicalBatchTopKSAE_singleTopK,
)

from dictionary_learning.dictionary_learning.trainers.hierarchical_gate import (
    HierarchicalSAE_Gated,
    HierarchicalSAEGatedTrainer,
)
from dictionary_learning.dictionary_learning.trainers.hierarchical_moe import (
    HierarchicalSAERecursiveTrainer,
    HierarchicalSAE_MOE,
)

from dictionary_learning.dictionary_learning.trainers.adaptive_k import (
    AdaptiveKTrainer,
    AdaptiveKSAE,
)
from dictionary_learning.dictionary_learning.trainers.res_hsae import (
    ResHSAETrainer,
    ResHSAE,
)

from dictionary_learning.dictionary_learning.trainers.priming_hsae import (
    PrimingHSAETrainer,
    PrimingHSAE,
)
from dictionary_learning.dictionary_learning.trainers.gr_sae import (
    GRSAETrainer,
    GRSAE,
)


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    MATRYOSHKA_BATCH_TOP_K = "matryoshka_batch_top_k"
    HIERARCHICAL_BATCH_TOP_K = "hierarchical_batch_top_k"
    HIERARCHICAL_BATCH_SINGLE_TOP_K = "hierarchical_batch_single_top_k"
    HIERARCHICAL_GATE = "hierarchical_gate"
    HIERARCHICAL_RECURSIVE = "hierarchical_recursive"
    ADAPTIVE_K = "adaptive_k"
    RES_HSAE = "res_hsae"
    PRIMING_HSAE = "priming_hsae"
    GR_SAE = "gr_sae"

@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


num_tokens = 500_000_000

print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [0]
dictionary_widths = [2**14, 2**16]
# dictionary_widths = [2**14]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8

learning_rates = [3e-4]
# learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

wandb_project = "qwen-32b-sweep"

LLM_CONFIG = {
    # "EleutherAI/pythia-70m-deduped": LLMConfig(
    #     llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    # ),
    # "EleutherAI/pythia-160m-deduped": LLMConfig(
    #     llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float32
    # # ),
    # "EleutherAI/pythia-160m-deduped": LLMConfig(
    #     llm_batch_size=32, context_length=2048, sae_batch_size=4096, dtype=t.float32
    # ),
    
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    # "EleutherAI/pythia-160m-deduped": LLMConfig(
    #     llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float32
    # ),
    "EleutherAI/pythia-160m-deduped": LLMConfig(
        llm_batch_size=32, context_length=2048, sae_batch_size=4096, dtype=t.bfloat16
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=4, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen2.5-Coder-32B-Instruct": LLMConfig(
        llm_batch_size=4, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    standard_new=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
)


TARGET_L0s = [80, 160]
# TARGET_L0s = [20, 40, 80, 160, 320, 640]
# TARGET_L0s = [20, 40, 80, 160]

ADAPTIVE_K_ALPHAS = [1e-5, 1e-4, 1e-3]
NUM_EXPERTS_LIST = [4, 8]
ROUTING_ALPHAS = [0.01, 0.1]
@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001

@dataclass
class HierarchicalBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    lower_level_latent_sizes: list[int] # 이름 및 타입 변경
    lower_level_ks: list[int]           # 이름 및 타입 변경
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000

@dataclass
class HierarchicalBatchTopKSAE_singleTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    lower_level_latent_sizes: list[int] # ks는 이제 필요 없습니다.
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000


@dataclass
class HierarchicalGateTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    lower_level_latent_sizes: list[int] # 이름 및 타입 변경
    lower_level_ks: list[int]           # 이름 및 타입 변경
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    
@dataclass
class HierarchicalRecursiveTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    lower_level_latent_sizes: list[int] # 이름 및 타입 변경
    lower_level_ks: list[int]           # 이름 및 타입 변경
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000

@dataclass
class AdaptiveKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k_predictor_alpha: float
    k_predictor_hidden_dim: int = 128
    
@dataclass
class ResHSAEConfig(HierarchicalBatchTopKTrainerConfig):
    pass

@dataclass
class PrimingHSAEConfig(HierarchicalBatchTopKTrainerConfig):
    context_dim: int = 64
    modulator_hidden_dim: int = 128

@dataclass
class GRSAEConfig(BaseTrainerConfig):
    dict_size: int  # Not used directly, but needed for base class
    seed: int
    lr: float
    l1_penalty: float
    num_experts: int
    expert_dict_size: int
    routing_loss_alpha: float


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.MATRYOSHKA_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
    if TrainerType.HIERARCHICAL_BATCH_TOP_K.value in architectures:
        # 테스트할 "하위 레벨" 구조들을 정의
        lower_level_structures = [
            {"lower_level_latent_sizes": [256], "lower_level_ks": [8]},
            # {"lower_level_latent_sizes": [32, 16], "lower_level_ks": [4, 4]},
        ]
        
        # 메인 sweep 루프: 전체 dict_size와 k, 그리고 하위 구조를 조합
        for seed, dict_size, learning_rate, k, structure in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(structure["lower_level_latent_sizes"]) if structure["lower_level_latent_sizes"] else 1
            prod_lower_ks = math.prod(structure["lower_level_ks"]) if structure["lower_level_ks"] else 1

            # 전체 파라미터가 하위 구조와 호환되는지 확인
            if dict_size % prod_lower_sizes != 0:
                continue
            if k % prod_lower_ks != 0:
                continue

            config = HierarchicalBatchTopKTrainerConfig(
                **base_config,
                trainer=HierarchicalBatchTopKTrainer,
                dict_class=HierarchicalBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=structure["lower_level_latent_sizes"],
                lower_level_ks=structure["lower_level_ks"],
                wandb_name=f"HierarchicalBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
            
    if TrainerType.HIERARCHICAL_BATCH_SINGLE_TOP_K.value in architectures:
        # 🚀 2. 테스트할 하위 레벨 구조를 sizes만 포함하도록 단순화
        TARGET_L0s = [64, 128]
        lower_level_structures = [
            [32, 16],  # 3-level 구조 # -> 64 / 
            [64],      # 2-level 구조 -> 256 cat
        ]
        
        # 🚀 3. 메인 루프에서 ks 관련 로직 제거
        for seed, dict_size, learning_rate, k, lower_sizes in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(lower_sizes) if lower_sizes else 1

            if dict_size % prod_lower_sizes != 0:
                continue
            
            # 🚀 4. Config 생성 시 ks 관련 인자 제거
            config = HierarchicalBatchTopKSAE_singleTopKTrainerConfig(
                **base_config,
                trainer=HierarchicalBatchTopKSAE_singleTopKTrainer,
                dict_class=HierarchicalBatchTopKSAE_singleTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=lower_sizes,
                wandb_name=f"HierarchicalBatchTopKSAE_singleTopK-{model_name}-{layer}",
            )
            trainer_configs.append(asdict(config))
    
    if TrainerType.HIERARCHICAL_BATCH_SINGLE_TOP_K_middleK.value in architectures:
        # 🚀 2. 테스트할 하위 레벨 구조를 sizes만 포함하도록 단순화
        TARGET_L0s = [64, 128] # -> final k
        lower_level_n = [8, 64, 256]

        # 🚀 3. 메인 루프에서 ks 관련 로직 제거
        for seed, dict_size, learning_rate, k, lower_sizes in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(lower_sizes) if lower_sizes else 1

            if dict_size % prod_lower_sizes != 0:
                continue
            
            # 🚀 4. Config 생성 시 ks 관련 인자 제거
            config = HierarchicalBatchTopKSAE_singleTopKTrainerConfig(
                **base_config,
                trainer=HierarchicalBatchTopKSAE_singleTopKTrainer,
                dict_class=HierarchicalBatchTopKSAE_singleTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=lower_sizes,
                wandb_name=f"HierarchicalBatchTopKSAE_singleTopK-{model_name}-{layer}",
            )
            trainer_configs.append(asdict(config))
            
    if TrainerType.HIERARCHICAL_GATE.value in architectures:
        # 테스트할 "하위 레벨" 구조들을 정의
        lower_level_structures = [
            {"lower_level_latent_sizes": [256], "lower_level_ks": [8]}, # -> 64 / 
            {"lower_level_latent_sizes": [32, 16], "lower_level_ks": [4, 4]},
        ]
        
        # 메인 sweep 루프: 전체 dict_size와 k, 그리고 하위 구조를 조합
        for seed, dict_size, learning_rate, k, structure in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(structure["lower_level_latent_sizes"]) if structure["lower_level_latent_sizes"] else 1
            prod_lower_ks = math.prod(structure["lower_level_ks"]) if structure["lower_level_ks"] else 1

            # 전체 파라미터가 하위 구조와 호환되는지 확인
            if dict_size % prod_lower_sizes != 0:
                continue
            if k % prod_lower_ks != 0:
                continue

            config = HierarchicalGateTrainerConfig(
                **base_config,
                trainer=HierarchicalSAEGatedTrainer,
                dict_class=HierarchicalSAE_Gated,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=structure["lower_level_latent_sizes"],
                lower_level_ks=structure["lower_level_ks"],
                wandb_name=f"HierarchicalSAEGatedTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
            
    if TrainerType.HIERARCHICAL_RECURSIVE.value in architectures:
        # 테스트할 "하위 레벨" 구조들을 정의
        lower_level_structures = [
            {"lower_level_latent_sizes": [256], "lower_level_ks": [8]},
            # {"lower_level_latent_sizes": [32, 16], "lower_level_ks": [4, 4]},
        ]
        
        # 메인 sweep 루프: 전체 dict_size와 k, 그리고 하위 구조를 조합
        for seed, dict_size, learning_rate, k, structure in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(structure["lower_level_latent_sizes"]) if structure["lower_level_latent_sizes"] else 1
            prod_lower_ks = math.prod(structure["lower_level_ks"]) if structure["lower_level_ks"] else 1

            # 전체 파라미터가 하위 구조와 호환되는지 확인
            if dict_size % prod_lower_sizes != 0:
                continue
            if k % prod_lower_ks != 0:
                continue

            config = HierarchicalRecursiveTrainerConfig(
                **base_config,
                trainer=HierarchicalSAERecursiveTrainer,
                dict_class=HierarchicalSAE_MOE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=structure["lower_level_latent_sizes"],
                lower_level_ks=structure["lower_level_ks"],
                wandb_name=f"HierarchicalSAE_Recursive-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
            
    if TrainerType.ADAPTIVE_K.value in architectures:
        for seed, dict_size, learning_rate, k_predictor_alpha in itertools.product(
            seeds, dict_sizes, learning_rates, ADAPTIVE_K_ALPHAS
        ):
            config = AdaptiveKTrainerConfig(
                **base_config,
                trainer=AdaptiveKTrainer,
                dict_class=AdaptiveKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k_predictor_alpha=k_predictor_alpha,
                wandb_name=f"AdaptiveKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.RES_HSAE.value in architectures:
        lower_level_structures = [
            {"lower_level_latent_sizes": [256], "lower_level_ks": [8]},
            {"lower_level_latent_sizes": [32, 16], "lower_level_ks": [4, 4]},
        ]
        for seed, dict_size, learning_rate, k, structure in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(structure["lower_level_latent_sizes"]) if structure["lower_level_latent_sizes"] else 1
            prod_lower_ks = math.prod(structure["lower_level_ks"]) if structure["lower_level_ks"] else 1
            if dict_size % prod_lower_sizes != 0:
                continue
            if k % prod_lower_ks != 0:
                continue
            config = ResHSAEConfig(
                **base_config,
                trainer=ResHSAETrainer,
                dict_class=ResHSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=structure["lower_level_latent_sizes"],
                lower_level_ks=structure["lower_level_ks"],
                wandb_name=f"ResHSAE-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))


    if TrainerType.GR_SAE.value in architectures:
        for seed, expert_dict_size, learning_rate, l1_penalty, num_experts, routing_alpha in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard, NUM_EXPERTS_LIST, ROUTING_ALPHAS
        ):
            config = GRSAEConfig(
                **base_config,
                trainer=GRSAETrainer,
                dict_class=GRSAE,
                lr=learning_rate,
                seed=seed,
                dict_size=0,
                l1_penalty=l1_penalty,
                num_experts=num_experts,
                expert_dict_size=expert_dict_size,
                routing_loss_alpha=routing_alpha,
                wandb_name=f"GRSAE-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.PRIMING_HSAE.value in architectures:
        lower_level_structures = [
            {"lower_level_latent_sizes": [256], "lower_level_ks": [8]},
            {"lower_level_latent_sizes": [32, 16], "lower_level_ks": [4, 4]},
        ]
        for seed, dict_size, learning_rate, k, structure in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, lower_level_structures
        ):
            prod_lower_sizes = math.prod(structure["lower_level_latent_sizes"]) if structure["lower_level_latent_sizes"] else 1
            prod_lower_ks = math.prod(structure["lower_level_ks"]) if structure["lower_level_ks"] else 1
            if dict_size % prod_lower_sizes != 0:
                continue
            if k % prod_lower_ks != 0:
                continue
            config = PrimingHSAEConfig(
                **base_config,
                trainer=PrimingHSAETrainer,
                dict_class=PrimingHSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                lower_level_latent_sizes=structure["lower_level_latent_sizes"],
                lower_level_ks=structure["lower_level_ks"],
                wandb_name=f"PrimingHSAE-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    return trainer_configs