"""STRGCN observation graph: joint time/variable relations, no Sandwich.

Retains the efficient layer's middle-stage Chebyshev operator and residual
update. All known variables can send to all valid variables at later/equal
physical times. Query nodes have no outgoing graph messages.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .TimeEmbedding import TimeEmbedding
from .STRGCN_Layer import masked_softmax
from .SparseRelations import RelationGroup, csr_block


@torch.no_grad()
def joint_sparse_plan(tt, valid, query):
    groups = []
    for b in range(len(tt)):
        src = torch.where(valid[b] & ~query[b])[0]
        dst = torch.where(valid[b])[0]
        if not len(src) or not len(dst):
            continue
        src = src[torch.argsort(tt[b, src], stable=True)]
        degree = torch.searchsorted(tt[b, src].contiguous(), tt[b, dst].contiguous(), right=True)
        active = degree > 0
        if active.any():
            groups.append(RelationGroup(b, 2, src[:int(degree.max())], dst[active], degree[active]))
    return groups


def time_features(encoder, tt, valid):
    p = encoder(tt, valid)
    return p.unsqueeze(0) if p.ndim == 2 else p


class DirectSTRGCNLayer(nn.Module):
    def __init__(self, dim, node_dim, num_vars, dropout, use_bias, use_norm,
                 time_embedding_type, gcn_type, gcn_depth, backend, chunk_size):
        super().__init__()
        self.gcn_type, self.gcn_depth = gcn_type, gcn_depth
        self.backend, self.chunk_size = backend, chunk_size
        self.time_embedding = TimeEmbedding(dim, time_embedding_type)
        self.learnable_adj_weight_U = nn.Embedding(num_vars, node_dim)
        self.learnable_adj_weight_V = nn.Embedding(num_vars, node_dim)
        self.middle_val_scale = nn.Parameter(torch.tensor(1.0))
        self.middle_time_scale = nn.Parameter(torch.tensor(1.0))
        self.middle_val_temp = nn.Parameter(torch.tensor(1.0))
        self.middle_time_temp = nn.Parameter(torch.tensor(1.0))
        self.middle_weight_T = nn.Linear(dim, dim)
        self.middle_weight_S_Embedding = nn.ModuleList([nn.Embedding(num_vars, dim) for _ in range(gcn_depth)])
        self.middle_bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        self.lambda_ = nn.Parameter(torch.tensor(.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

    def support_apply(self, x, p, vars_idx, tt, valid, query, plan):
        u, v = self.learnable_adj_weight_U(vars_idx), self.learnable_adj_weight_V(vars_idx)
        time_scale = self.middle_time_temp * self.middle_time_scale
        var_scale = self.middle_val_temp * self.middle_val_scale
        if self.backend == 'sparse_csr':
            messages = [torch.zeros_like(row) for row in x]
            for group in plan:
                b, src, dst = group.batch, group.sources, group.receivers
                pj, value, vj = p[b, src], x[b, src], v[b, src]
                for start in range(0, len(dst), self.chunk_size):
                    nodes = dst[start:start+self.chunk_size]
                    degrees = group.degrees[start:start+self.chunk_size]
                    def aggregate(pi, pj, value, ui, vj, degree=degrees):
                        return csr_block(pi, pj, value, ui, vj, degree,
                                         time_scale, var_scale, True, variable_normalizer=1.0)
                    args = (p[b, nodes], pj, value, u[b, nodes], vj)
                    msg = checkpoint(aggregate, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else aggregate(*args)
                    messages[b] = messages[b].index_add(0, nodes, msg)
            # Dense empty-neighborhood execution produces explicit zero gradients.
            zero = (p.sum()+u.sum()+v.sum()+x.sum()+time_scale+var_scale)*0
            return torch.stack(messages)+zero
        blocks = []
        for start in range(0, x.shape[1], self.chunk_size):
            end = min(start+self.chunk_size, x.shape[1])
            def aggregate(p, x, u, v, start=start, end=end):
                logits = time_scale * torch.einsum('bid,bjd->bij', p[:,start:end], p)
                logits = logits + var_scale * torch.einsum('bid,bjd->bij', u[:,start:end], v)
                mask = (valid[:,start:end,None] & (valid & ~query)[:,None,:]
                        & (tt[:,None,:] <= tt[:,start:end,None]))
                return torch.bmm(masked_softmax(logits, mask), x)
            args = (p,x,u,v)
            blocks.append(checkpoint(aggregate,*args,use_reentrant=False) if self.training and torch.is_grad_enabled() else aggregate(*args))
        return torch.cat(blocks,1)

    def forward(self, x, vars_idx, tt, valid, query, plan):
        p = time_features(self.time_embedding, tt, valid)
        weight_t = self.middle_weight_T(p)
        if self.gcn_type == 'gcn':
            # Original low_rank_graph_conv: source transform BEFORE propagation.
            source = (x * weight_t).sum(-1,keepdim=True) * self.middle_weight_S_Embedding[0](vars_idx)
            out = self.support_apply(source,p,vars_idx,tt,valid,query,plan)
        else:
            # Original low_rank_cheb_graph_conv: propagate T_k(A)H, THEN
            # apply the receiver's time/variable factor for each order k.
            prev2 = x
            out = (prev2*weight_t).sum(-1,keepdim=True)*self.middle_weight_S_Embedding[0](vars_idx)
            if self.gcn_depth > 1:
                prev1 = self.support_apply(x,p,vars_idx,tt,valid,query,plan)
                out = out + (prev1*weight_t).sum(-1,keepdim=True)*self.middle_weight_S_Embedding[1](vars_idx)
            for k in range(2,self.gcn_depth):
                current = 2*self.support_apply(prev1,p,vars_idx,tt,valid,query,plan)-prev2
                out = out + (current*weight_t).sum(-1,keepdim=True)*self.middle_weight_S_Embedding[k](vars_idx)
                prev2,prev1 = prev1,current
        if self.middle_bias is not None:
            out = out+self.middle_bias
        out = self.norm(self.lambda_*self.dropout(F.relu(out))+(1-self.lambda_)*x)
        return torch.where(valid.unsqueeze(-1),out,0)


class STRGCNDirect(nn.Module):
    def __init__(self, c_in=1,c_out=1,hid_dim=16,n_layer=3,node_dim=16,dropout=0.,
                 use_bias=True,use_norm=True,time_embedding_type='rbf',gcn_type='cheb',gcn_depth=2,
                 relation_backend='sparse_csr',receiver_chunk_size=128,num_vars=None,global_args=None,
                 use_efficient_gcn=False,use_info_agg=False,use_revin=False,**kwargs):
        super().__init__()
        if c_in!=1 or c_out!=1 or hid_dim%2:
            raise ValueError('Use scalar nodes and an even hid_dim')
        if use_efficient_gcn or use_info_agg or use_revin:
            raise ValueError('STRGCNDirect excludes Sandwich, aggregation and RevIN')
        if gcn_type not in ('cheb','gcn') or not isinstance(gcn_depth,int) or gcn_depth<1:
            raise ValueError('Use cheb with positive gcn_depth, or gcn with gcn_depth=1')
        if gcn_type=='gcn' and gcn_depth!=1:
            raise ValueError('gcn requires gcn_depth=1')
        if relation_backend not in ('dense','sparse_csr') or not isinstance(receiver_chunk_size,int) or receiver_chunk_size<1 or n_layer<1:
            raise ValueError('Invalid relation backend, chunk size or layer depth')
        if time_embedding_type not in ('rbf','rope','transformer'):
            raise ValueError('Unsupported time embedding')
        self.num_vars = num_vars or (global_args or {}).get('num_of_vals')
        if not self.num_vars:
            raise ValueError('Supply num_vars or global_args[num_of_vals]')
        self.relation_backend = relation_backend
        self.encoder = nn.Linear(c_in,hid_dim)
        self.pred_token = nn.Parameter(torch.randn(1,1,hid_dim))
        self.time_embedding = TimeEmbedding(hid_dim,time_embedding_type)
        self.layers = nn.ModuleList([DirectSTRGCNLayer(hid_dim,node_dim,self.num_vars,dropout,use_bias,use_norm,
            time_embedding_type,gcn_type,gcn_depth,relation_backend,receiver_chunk_size) for _ in range(n_layer)])
        self.lambda_ = nn.Parameter(torch.tensor(.5))
        self.projection = nn.Linear(hid_dim,c_out)

    def forward(self,batch_value,batch_timestamp,batch_var_idx,batch_pred_mask=None,
                batch_pad_mask=None,batch_time_id=None,batch_agg_mask=None,batch_node_type=None):
        if batch_value.ndim!=2 or batch_value.shape[1]==0 or batch_pred_mask is None:
            raise ValueError('Supply nonempty [B,L] node tensors and batch_pred_mask')
        valid = torch.ones_like(batch_value,dtype=torch.bool) if batch_pad_mask is None else batch_pad_mask.bool()
        if any(v.shape!=batch_value.shape for v in (batch_timestamp,batch_var_idx,batch_pred_mask,valid)):
            raise ValueError('Node tensors must share [B,L] shape')
        chosen = batch_var_idx[valid]
        if not torch.isfinite(chosen).all() or (chosen!=chosen.long()).any() or ((chosen<0)|(chosen>=self.num_vars)).any():
            raise ValueError('Invalid variable IDs')
        ids = torch.where(valid,batch_var_idx,0).long()
        query = batch_pred_mask.bool() & valid
        tt = torch.where(valid,batch_timestamp,0)
        value = torch.where(valid & ~query,batch_value,0)
        if not torch.isfinite(tt).all() or not torch.isfinite(value).all():
            raise ValueError('Known values and valid times must be finite')
        h0 = self.encoder(value.unsqueeze(-1))+time_features(self.time_embedding,tt,valid)
        h0 = torch.where(query.unsqueeze(-1),self.pred_token,h0)
        h0 = torch.where(valid.unsqueeze(-1),h0,0)
        plan = joint_sparse_plan(tt,valid,query) if self.relation_backend=='sparse_csr' else None
        h = h0
        for layer in self.layers:
            h = layer(h,ids,tt,valid,query,plan)
        out = self.projection(self.lambda_*h+(1-self.lambda_)*h0)
        return torch.where(valid.unsqueeze(-1),out,0)
