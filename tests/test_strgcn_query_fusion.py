"""STRGCN operator parity and unrestricted-X information flow."""

import copy

import pytest
import torch
from torch.nn import functional as F

from models.STRGCN.STRGCN_Layer import low_rank_cheb_graph_conv, masked_softmax
from models.STRGCN.STRGCNDirect import time_features
from models.exogenous.fusion import DynamicLagFusion, Observations
from models.exogenous.strgcn_encoder import UnrestrictedSTRGCNLayer
from test_query_fusion import prepared, toy
from test_tpatch_records import make_data


def test_original_strgcn_middle_operator_value_and_gradient_parity():
    torch.manual_seed(7)
    layer = UnrestrictedSTRGCNLayer(
        8, 3, 3, 0.0, True, True, "rbf", "cheb", 3, "dense", 2
    ).double()
    x = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    tt = torch.tensor([[0.8, 0.1, 0.5, 0.9], [0.2, 0.6, 0.1, 0.7]], dtype=torch.double)
    ids = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool()
    query = torch.zeros_like(valid)
    p = time_features(layer.time_embedding, tt, valid)
    scores = (
        layer.middle_time_temp * layer.middle_time_scale * (p @ p.transpose(-1, -2))
    )
    scores = scores + layer.middle_val_temp * layer.middle_val_scale * (
        layer.learnable_adj_weight_U(ids)
        @ layer.learnable_adj_weight_V(ids).transpose(-1, -2)
    )
    a = masked_softmax(scores, valid[:, :, None] & valid[:, None, :])
    expected = torch.cat(
        [
            low_rank_cheb_graph_conv(
                x[b : b + 1],
                a[b : b + 1],
                layer.middle_weight_T(p[b : b + 1]),
                torch.stack([emb(ids[b]) for emb in layer.middle_weight_S_Embedding]),
                layer.gcn_depth,
                layer.middle_bias,
            )
            for b in range(len(x))
        ]
    )
    expected = layer.norm(layer.lambda_ * F.relu(expected) + (1 - layer.lambda_) * x)
    expected = torch.where(valid[..., None], expected, 0.0)
    actual = layer(x, ids, tt, valid, query, None)
    torch.testing.assert_close(actual, expected)
    params = (x, layer.learnable_adj_weight_U.weight, layer.middle_weight_T.weight)
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), params)
    for aa, ee in zip(ga, ge):
        torch.testing.assert_close(aa, ee)


def test_context_future_x_padding_order_chunking_and_empty_identity():
    x, _ = toy()
    m = (
        DynamicLagFusion(
            6,
            2,
            hidden=8,
            heads=2,
            x_encoder="strgcn",
            allow_future_x=True,
            chunk_size=1,
        )
        .double()
        .eval()
    )
    x.values = x.values.double().requires_grad_()
    x.times = x.times.double()
    f = torch.randn(1, 1, 6, dtype=torch.double)
    t = torch.tensor([[0.05]], dtype=torch.double)  # ALL supplied X are later
    ids, valid = torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool)
    before = m(f, t, ids, valid, x)
    grad = torch.autograd.grad(before.square().sum(), x.values)[0]
    assert (grad.abs() > 1e-10).all()
    a = m.x_context(m._clean(x))
    changed = copy.deepcopy(
        Observations(x.values.detach().clone(), x.times, x.variables, x.valid)
    )
    changed.values[:, -1] += 10
    assert not torch.allclose(a[:, 0], m.x_context(m._clean(changed))[:, 0])
    # No query-time restriction either in X encoding or in final fusion.
    assert not torch.allclose(before, m(f, t, ids, valid, changed))
    order = torch.tensor([2, 0, 1])
    permuted = Observations(
        *(
            getattr(x, name)[:, order]
            for name in ("values", "times", "variables", "valid")
        )
    )
    torch.testing.assert_close(before, m(f, t, ids, valid, permuted))
    m.chunk_size = 100
    m.x_context.layers[0].chunk_size = 100
    torch.testing.assert_close(before, m(f, t, ids, valid, x))
    padded = Observations(
        torch.cat([x.values, torch.tensor([[float("nan")]], dtype=torch.double)], 1),
        torch.cat([x.times, torch.tensor([[float("nan")]], dtype=torch.double)], 1),
        torch.cat([x.variables, torch.tensor([[999]])], 1),
        torch.tensor([[1, 1, 1, 0]]).bool(),
    )
    torch.testing.assert_close(before, m(f, t, ids, valid, padded))
    padded.valid[:] = False
    torch.testing.assert_close(f, m(f, t, ids, valid, padded))


@pytest.mark.parametrize("backbone", ["tpatchgnn", "hipatch", "mtand"])
@pytest.mark.parametrize("dataset", ["activity", "mimic", "physionet", "ushcn"])
def test_backbones_labels_future_x_and_backward(tmp_path, backbone, dataset):
    from models.exogenous.adapter import TPatchQueryFusion, HiPatchQueryFusion
    from models.exogenous.mtan_adapter import MTANQueryFusion
    from test_mtan_query_fusion import prepared as mtan_prepared

    torch.manual_seed(12)
    make_data(tmp_path, dataset)
    g, ds = (mtan_prepared if backbone == "mtand" else prepared)(tmp_path, dataset)
    sample, origin = ds[0]
    b = ds.collate_fn([(sample, origin)])
    params = dict(
        hid_dim=8,
        n_heads=1,
        plugin_hidden=8,
        plugin_heads=2,
        fusion_mode="v1",
        x_encoder="strgcn",
        allow_future_x=True,
    )
    if backbone == "mtand":
        params.update(latent_dim=4, gen_hidden=8, te_dim=8, num_ref_points=8)
        cls = MTANQueryFusion
    else:
        params["n_layer"] = 1
        cls = TPatchQueryFusion if backbone == "tpatchgnn" else HiPatchQueryFusion
    m = cls(g, **params).eval()
    before = m(**b)
    altered = sample.clone()
    altered[b["batch_pred_mask"][0].bool(), 1] = float("nan")
    torch.testing.assert_close(before, m(**ds.collate_fn([(altered, origin)])))
    future = torch.where(
        (b["batch_node_type"][0] == 2) & b["batch_pad_mask"][0].bool()
    )[0]
    if len(future):
        ix = future[b["batch_timestamp"][0, future].argmax()]
        earlier = b["batch_pred_mask"].bool() & (
            b["batch_timestamp"] < b["batch_timestamp"][0, ix]
        )
        if earlier.any():
            altered = sample.clone()
            altered[ix, 1] += 100
            changed = ds.collate_fn([(altered, origin)])
            features = m.encode_queries(b)
            query_earlier = b["batch_backbone_queries"] < b["batch_timestamp"][0, ix]
            assert not torch.allclose(
                m.fuse(features, b)[query_earlier],
                m.fuse(features, changed)[query_earlier],
                atol=1e-8,
                rtol=1e-7,
            )
    m.train()
    prediction = m(**b)
    prediction.square().mean().backward()
    layer = m.fusion.x_context.layers[0]
    assert layer.learnable_adj_weight_U.weight.grad.abs().sum() > 0
    assert layer.middle_weight_T.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    assert g["query_fusion_protocol"]["x_visibility"] == "all supplied observed X"


def test_reject_context_hidden_causal_leak():
    with pytest.raises(ValueError, match="allow_future_x"):
        DynamicLagFusion(6, 2, x_encoder="strgcn")
