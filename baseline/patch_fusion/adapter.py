"""Original patch backbones plus DAG's pointwise two-layer MLP Fusion."""
from types import SimpleNamespace
import torch
from torch import nn


class DAGMLPFusion(nn.Module):
    """decisionintelligence/DAG baselines/utils.py::MLP, without residual."""
    def __init__(self, targets, exogenous, hidden=64):
        super().__init__()
        self.linear1 = nn.Linear(targets + exogenous, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, targets)

    def forward(self, future_x, output):
        return self.linear2(self.relu(self.linear1(torch.cat((output.float(), future_x.float()), -1))))


class PatchFusion(nn.Module):
    def __init__(self, global_args, backbone, hid_dim, n_layer=1, n_heads=1,
                 fusion_hidden=64, te_dim=10, node_dim=10, hop=1, alpha=1., res=1.):
        super().__init__()
        self.num_vars = global_args['num_of_vals']
        self.target_ids = list(global_args['fusion_target_ids'])
        self.exogenous_ids = list(global_args['fusion_exogenous_ids'])
        self.backbone_ids = list(global_args['fusion_backbone_ids'])
        self.register_buffer('target_column', torch.full((self.num_vars,), -1, dtype=torch.long))
        self.target_column[self.target_ids] = torch.arange(len(self.target_ids))
        patches = global_args['fusion_num_patches']
        if min(hid_dim, n_layer, n_heads, fusion_hidden, te_dim, node_dim, hop) < 1 or hid_dim % n_heads:
            raise ValueError('Invalid backbone or fusion dimensions')
        args = SimpleNamespace(device=torch.device(global_args.get('device', 'cpu')),
            ndim=len(self.backbone_ids), npatch=patches, hid_dim=hid_dim, nlayer=n_layer,
            nhead=n_heads, tf_layer=1, te_dim=te_dim, node_dim=node_dim, hop=hop,
            outlayer='Linear', task='forecasting', alpha=alpha, res=res,
            patch_layer=(patches - 1).bit_length() + 1)
        if backbone == 'tpatchgnn':
            from .vendor.tpatchgnn import tPatchGNN
            self.core = tPatchGNN(args)
        elif backbone == 'hipatch':
            from .vendor.hipatch import Hi_Patch
            self.core = Hi_Patch(args)
        else:
            raise ValueError('Unsupported patch backbone')
        self.fusion = DAGMLPFusion(len(self.target_ids), len(self.exogenous_ids), fusion_hidden)

    def forward(self, batch_value, batch_timestamp, batch_var_idx, batch_pred_mask,
                batch_pad_mask, batch_node_type, batch_backbone_values,
                batch_backbone_times, batch_backbone_mask, batch_backbone_queries,
                batch_backbone_future_x, batch_backbone_query_slot, **unused):
        base = self.core.forecasting(batch_backbone_queries, batch_backbone_values,
                                     batch_backbone_times, batch_backbone_mask)
        base = base.reshape(1, batch_value.shape[0], batch_backbone_queries.shape[1], len(self.backbone_ids))[0]
        z = base[..., :len(self.target_ids)]
        prediction = self.fusion(batch_backbone_future_x, z)
        column = self.target_column[batch_var_idx.long()].clamp_min(0)
        B = batch_value.shape[0]
        output = prediction[torch.arange(B, device=prediction.device)[:, None],
                            batch_backbone_query_slot, column]
        # Queries are the only outputs consumed by the common trainer.
        query = batch_pred_mask.bool() & batch_pad_mask.bool()
        return torch.where(query, output, 0.).unsqueeze(-1)


class TPatchGNNFusion(PatchFusion):
    def __init__(self, global_args, **kwargs):
        super().__init__(global_args, backbone='tpatchgnn', **kwargs)


class HiPatchFusion(PatchFusion):
    def __init__(self, global_args, **kwargs):
        super().__init__(global_args, backbone='hipatch', **kwargs)
