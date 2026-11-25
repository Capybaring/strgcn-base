"""Four equal-width blocks go directly to attention; density ablation stays matched."""

import copy

import pytest
import torch

from models.STRGCN.STRGCNDirect import time_features
from models.exogenous.voronoi_graph_transformer import VoronoiGraphTransformer
from test_irregular_graph_transformer import inputs
from test_voronoi_graph_transformer import fusion


@pytest.mark.parametrize("layers", [1, 2])
def test_equal_width_channels_enter_qkv_without_fusion_mlp(layers):
    m = (
        fusion(density_location="node", node_encoding="concat", x_layers=layers)
        .double()
        .eval()
    )
    x, *_ = inputs()
    core = m.x_context
    seen = {}

    def before_attention(module, args):
        seen["h"] = args[0]

    def before_query(module, args):
        seen["q_input"] = args[0]

    hooks = [
        core.layers[0].register_forward_pre_hook(before_attention),
        core.layers[0].q.register_forward_pre_hook(before_query),
    ]
    out = core(x)
    for hook in hooks:
        hook.remove()
    d = m.hidden
    assert out.shape == (*x.values.shape, d)
    assert seen["h"].shape[-1] == 4 * d
    assert not hasattr(core, "node_fusion")
    assert core.density_encoder.in_features == 1
    assert core.density_encoder.out_features == d
    assert core.density_encoder.bias is None
    assert not any(isinstance(module, torch.nn.SiLU) for module in core.modules())
    from models.exogenous.voronoi_graph_transformer import voronoi_log_density

    s = voronoi_log_density(x.times, x.variables, x.valid)
    expected = [
        core.encoder(x.values[..., None]),
        core.variable(x.variables),
        time_features(core.time_embedding, x.times, x.valid),
        core.density_encoder(s[..., None]),
    ]
    for actual, block in zip(seen["h"].split(d, -1), expected):
        torch.testing.assert_close(actual, block, atol=0, rtol=0)
    torch.testing.assert_close(
        seen["q_input"], core.layers[0].norm1(seen["h"]), atol=0, rtol=0
    )
    for projection in (
        core.layers[0].q,
        core.layers[0].k,
        core.layers[0].v,
        core.layers[0].residual,
    ):
        assert projection.in_features == 4 * d and projection.out_features == d
    if layers > 1:
        assert core.layers[1].q.in_features == d
        assert isinstance(core.layers[1].residual, torch.nn.Identity)
    protocol = m.protocol()
    assert protocol["node_input_dim"] == 4 * d
    assert protocol["node_channel_dim"] == protocol["attention_hidden_dim"] == d
    assert not protocol["pre_attention_fusion_mlp"]
    assert (
        protocol["density_node_encoding"] and not protocol["density_attention_encoding"]
    )


def test_density_ablation_zeros_whole_block_and_is_independent_of_encoder_weights():
    m = fusion(density_location="node", node_encoding="concat").double().eval()
    plain = (
        fusion(
            density_location="node", node_encoding="concat", use_density_encoding=False
        )
        .double()
        .eval()
    )
    plain.load_state_dict(m.state_dict(), strict=True)
    assert sum(p.numel() for p in m.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    x, f, t, ids, valid = inputs()
    captured = []
    hook = plain.x_context.layers[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    before = plain(f, t, ids, valid, x)
    assert torch.count_nonzero(captured[0][..., -m.hidden :]) == 0
    with torch.no_grad():
        plain.x_context.density_encoder.weight.fill_(123.0)
    torch.testing.assert_close(before, plain(f, t, ids, valid, x), atol=0, rtol=0)
    before.square().sum().backward()
    assert torch.count_nonzero(plain.x_context.density_encoder.weight.grad) == 0
    hook.remove()
    assert not torch.allclose(m(f, t, ids, valid, x), before, atol=1e-10, rtol=1e-10)
    regular = copy.deepcopy(x)
    regular.times = x.times.new_tensor([[0.1, 0.3, 0.5]])
    assert (
        torch.count_nonzero(m.x_context.density_encoder(regular.times.new_zeros(3, 1)))
        == 0
    )
    torch.testing.assert_close(
        m(f, t, ids, valid, regular), plain(f, t, ids, valid, regular), atol=0, rtol=0
    )


@pytest.mark.parametrize("name", ["grafiti", "cru", "latentode"])
@pytest.mark.parametrize("compare", [False, True])
def test_four_dimensional_launcher_new_backbones(tmp_path, name, compare):
    from test_voronoi_graph_transformer import (
        test_three_batches_cuda_default_configs_override_and_no_overwrite,
    )

    test_three_batches_cuda_default_configs_override_and_no_overwrite(
        tmp_path, name, compare, "voronoi_h_4d"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(node_encoding="unknown"),
        dict(node_encoding="concat", density_location="attention"),
    ],
)
def test_invalid_node_encoding_fails(kwargs):
    with pytest.raises(ValueError):
        VoronoiGraphTransformer(8, 3, heads=2, **kwargs)
