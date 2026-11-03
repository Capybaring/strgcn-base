"""Original ExEn roles/stages with the STRGCNDirect Chebyshev operator."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .ExEn import ExEn, relation_masks
from .TimeEmbedding import TimeEmbedding
from .STRGCN_Layer import masked_softmax
from .SparseRelations import csr_block
from .STRGCNDirect import receiver_cheb_convolution, time_features


class AlignedRelationLayer(nn.Module):
    """Same three relation domains as ExEn; receiver-side Chebyshev factors.

    A temporal layer has separate X/Y relation parameters and one shared layer
    time encoder. A cross layer updates Y only. Its T0 self term remains active
    without X neighbors, so the Y-only ablation still executes every layer.
    """
    def __init__(self, dim, node_dim, num_vars, dropout, use_bias, use_norm,
                 cross=False, depth=2, time_embedding_type='rbf'):
        super().__init__()
        self.cross, self.depth = cross, depth
        count = 1 if cross else 2
        self.time_embedding = TimeEmbedding(dim, time_embedding_type)
        self.weight_T = nn.ModuleList([nn.Linear(dim, dim) for _ in range(count)])
        self.weight_S = nn.ModuleList([
            nn.ModuleList([nn.Embedding(num_vars, dim) for _ in range(depth)])
            for _ in range(count)])
        self.time_scale = nn.Parameter(torch.ones(count))
        self.time_temp = nn.Parameter(torch.ones(count))
        if cross:
            self.adj_U = nn.Embedding(num_vars, node_dim)
            self.adj_V = nn.Embedding(num_vars, node_dim)
            self.var_scale = nn.Parameter(torch.tensor(1.0))
            self.var_temp = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        self.lambda_ = nn.Parameter(torch.tensor(.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

    def _support(self, x, p, var_idx, masks, valid, tt, types, plan, chunk, index):
        time_scale = self.time_scale[index] * self.time_temp[index]
        if self.cross:
            u, v = self.adj_U(var_idx), self.adj_V(var_idx)
            var_scale = self.var_scale * self.var_temp
        else:
            u, v = p[..., :0], p[..., :0]
            var_scale = time_scale.new_zeros(())
        if plan is not None:
            messages = [torch.zeros_like(row) for row in x]
            relation = 2 if self.cross else index
            for group in plan:
                if group.relation != relation:
                    continue
                b, src, dst = group.batch, group.sources, group.receivers
                for start in range(0, len(dst), chunk):
                    nodes = dst[start:start+chunk]
                    degrees = group.degrees[start:start+chunk]
                    def aggregate(pi, pj, value, ui, vj, degree=degrees):
                        return csr_block(pi, pj, value, ui, vj, degree,
                                         time_scale, var_scale, self.cross, variable_normalizer=1.0)
                    args = (p[b,nodes], p[b,src], x[b,src], u[b,nodes], v[b,src])
                    msg = checkpoint(aggregate,*args,use_reentrant=False) if self.training and torch.is_grad_enabled() else aggregate(*args)
                    messages[b] = messages[b].index_add(0,nodes,msg)
            zero = (p.sum()+x.sum()+u.sum()+v.sum()+time_scale+var_scale)*0
            return torch.stack(messages)+zero
        blocks = []
        for start in range(0,x.shape[1],chunk):
            end = min(start+chunk,x.shape[1])
            if masks is None:
                all_masks = relation_masks(tt,var_idx,types,valid,slice(start,end))
                mask = all_masks[2 if self.cross else index]
            else:
                mask = masks[index][:,start:end]
            def aggregate(p,x,u,v,start=start,end=end,mask=mask):
                logits = time_scale*torch.einsum('bid,bjd->bij',p[:,start:end],p)
                if self.cross:
                    logits = logits+var_scale*torch.einsum('bid,bjd->bij',u[:,start:end],v)
                return torch.bmm(masked_softmax(logits,mask),x)
            args = (p,x,u,v)
            blocks.append(checkpoint(aggregate,*args,use_reentrant=False) if self.training and torch.is_grad_enabled() else aggregate(*args))
        return torch.cat(blocks,1)

    def forward(self,h,p,var_idx,masks,valid,timestamp=None,node_type=None,
                receiver_chunk_size=None,sparse_plan=None):
        # Each layer owns its time features, exactly as STRGCNDirect does.
        p = time_features(self.time_embedding,timestamp,valid)
        chunk = receiver_chunk_size or 128
        exo = (node_type==0)|(node_type==2)
        message = torch.zeros_like(h)
        for index in range(len(self.weight_T)):
            apply_support = lambda value,index=index: self._support(
                value,p,var_idx,masks,valid,timestamp,node_type,sparse_plan,chunk,index)
            branch = receiver_cheb_convolution(h,apply_support,self.weight_T[index](p),
                                              [embedding(var_idx) for embedding in self.weight_S[index]])
            domain = valid & (~exo if self.cross or index==1 else exo)
            message = message+torch.where(domain.unsqueeze(-1),branch,0)
        if self.bias is not None:
            message = message+self.bias
        out = self.norm(self.lambda_*self.dropout(F.relu(message))+(1-self.lambda_)*h)
        if self.cross:
            # X remains fixed in this stage; Y retains the Chebyshev T0 update
            # even when no exogenous observations are present.
            out = torch.where((valid & ~exo).unsqueeze(-1),out,h)
        return torch.where(valid.unsqueeze(-1),out,0)


class ExEnAligned(ExEn):
    """Retain ExEn input/role validation, role/status embeddings and 2+1 stages."""
    def __init__(self, *args, gcn_type='cheb', gcn_depth=2,
                 time_embedding_type='rbf', **kwargs):
        self._layer_time_embedding_type = time_embedding_type
        super().__init__(*args,gcn_type=gcn_type,gcn_depth=gcn_depth,
                         time_embedding_type=time_embedding_type,**kwargs)

    def _validate_operator(self,gcn_type,gcn_depth):
        if gcn_type!='cheb' or not isinstance(gcn_depth,int) or gcn_depth<1:
            raise ValueError('ExEnAligned requires cheb with positive gcn_depth (default 2)')
        self.gcn_depth = gcn_depth

    def _make_relation_layer(self,*args,cross=False):
        return AlignedRelationLayer(*args,cross=cross,depth=self.gcn_depth,
                                    time_embedding_type=self._layer_time_embedding_type)
