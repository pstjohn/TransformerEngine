import pytest
import torch
from transformer_engine.pytorch.transformer import TransformerLayer
from transformers import PretrainedConfig, PreTrainedModel

from tests.pytorch.test_sanity import custom_amax_compute, custom_amax_to_scale
from transformer_engine.common import recipe
from transformer_engine.pytorch.utils import is_bf16_compatible


def custom_amax_to_scale(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: torch.Tensor,
    recipe: recipe.DelayedScaling,
) -> torch.Tensor:
    """Custom func to test recipe."""
    sf = fp8_max / amax
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)

    return sf


def custom_amax_compute(amax_history: torch.Tensor) -> torch.Tensor:
    """Custom func to test recipe."""
    return torch.min(amax_history, dim=0).values


fp8_recipes = [
    None,  # Test non-FP8
    recipe.MXFP8BlockScaling(),  # Test default
    recipe.Float8CurrentScaling(),  # Test default
    recipe.Float8BlockScaling(),  # Test default
    recipe.DelayedScaling(),  # Test default
    recipe.DelayedScaling(  # Test most_recent algo
        amax_history_len=16,
        amax_compute_algo="most_recent",
    ),
    recipe.DelayedScaling(  # Test custom amax and scale compute algo
        fp8_format=recipe.Format.E4M3,
        amax_compute_algo=custom_amax_compute,
        scaling_factor_compute_algo=custom_amax_to_scale,
    ),
]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]
batch_sizes_with_zero = [0, 1, 2]

all_activations = [
    "gelu",
    "relu",
    "reglu",
    "geglu",
    "swiglu",
    "srelu",
    "qgelu",
    "qgeglu",
]
all_normalizations = ["LayerNorm", "RMSNorm"]


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_save_and_load_hf_model(tmp_path, dtype, activation, normalization):
    class SimpleTEModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config: PretrainedConfig):
            super().__init__(config)
            self.my_layer = TransformerLayer(
                hidden_size=320,
                num_attention_heads=16,
                ffn_hidden_size=1024,
                layer_number=None,
                params_dtype=dtype,
                activation=activation,
                normalization=normalization,
            )

        def forward(self, hidden_states, attention_mask):
            return self.my_layer(hidden_states, attention_mask)

    model = SimpleTEModel(PretrainedConfig())
    model.save_pretrained(tmp_path / "simple_te_model")
    del model
    SimpleTEModel.from_pretrained(tmp_path / "simple_te_model")
