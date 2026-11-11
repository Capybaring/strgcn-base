"""Content attention and graph biases with independent time/variable stages.

Sorted legal source groups are shared with ExEn. No CSR tensor or graph
convolution is used. Low-rank time/variable biases are folded into Q/K;
Legacy execution streams receiver groups; packed execution batches size buckets.
Only a bounded bucket's physical-time mask is materialized.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .ExEn import ExEn
from .SparseRelations import build_sparse_relations, build_within_variable_relations
from .STRGCNDirect import time_features
from .TimeEmbedding import TimeEmbedding
from .PackedAttention import PackedAttentionPlan, pack_attention_plan


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, node_dim, num_vars, dropout, use_bias, use_norm,
                 cross=False, num_heads=2, ffn_multiplier=2,
                 time_embedding_type='rbf', attention_backend='sdpa',
                 attention_dtype='float32', attention_dropout=0.,
                 use_time_bias=True, use_variable_bias=True,
                 time_bias_init=1., variable_bias_init=.01, checkpoint_attention=True, relations=None):
        super().__init__()
        if not isinstance(num_heads,int) or num_heads<1 or dim%num_heads:
            raise ValueError('num_heads must be positive and divide hid_dim')
        if not isinstance(ffn_multiplier,int) or ffn_multiplier<1:
            raise ValueError('ffn_multiplier must be a positive integer')
        if attention_backend not in ('sdpa','reference'):
            raise ValueError('attention_backend must be sdpa or reference')
        if attention_dtype not in ('float32','auto','bfloat16','float16'):
            raise ValueError('Unsupported attention_dtype')
        if not 0<=attention_dropout<1 or not math.isfinite(time_bias_init) or not math.isfinite(variable_bias_init):
            raise ValueError('Invalid attention dropout or bias initialization')
        if not isinstance(checkpoint_attention, bool):
            raise ValueError('checkpoint_attention must be boolean')
        self.checkpoint_attention = checkpoint_attention
        self.cross,self.num_heads,self.head_dim = cross,num_heads,dim//num_heads
        self.relation_ids = tuple(relations) if relations is not None else ((2,) if cross else (0, 1))
        self.within_group = relations is not None
        self.backend,self.attention_dtype = attention_backend,attention_dtype
        self.attention_dropout = attention_dropout
        self.use_time_bias,self.use_variable_bias = use_time_bias,use_variable_bias and (cross or self.within_group)
        count = len(self.relation_ids)
        self.norm_attention = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.norm_ffn = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.q_proj=nn.ModuleList([nn.Linear(dim,dim,bias=use_bias) for _ in range(count)])
        self.k_proj=nn.ModuleList([nn.Linear(dim,dim,bias=use_bias) for _ in range(count)])
        self.v_proj=nn.ModuleList([nn.Linear(dim,dim,bias=use_bias) for _ in range(count)])
        self.out_proj=nn.ModuleList([nn.Linear(dim,dim,bias=use_bias) for _ in range(count)])
        self.time_embedding=TimeEmbedding(dim,time_embedding_type) if use_time_bias else None
        if use_time_bias:
            self.time_strength=nn.Parameter(torch.full((count,num_heads),float(time_bias_init)))
        if self.use_variable_bias:
            self.adj_U=nn.Embedding(num_vars,num_heads*node_dim)
            self.adj_V=nn.Embedding(num_vars,num_heads*node_dim)
            self.variable_strength=nn.Parameter(torch.full((num_heads,),float(variable_bias_init)))
        self.ffn=nn.Sequential(nn.Linear(dim,dim*ffn_multiplier,bias=use_bias),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(dim*ffn_multiplier,dim,bias=use_bias))
        self.dropout=nn.Dropout(dropout)

    def _heads(self,x):
        return x.reshape(*x.shape[:-1],self.num_heads,self.head_dim).transpose(1,2)

    def _qkv(self,h,p,ids,index):
        # scale=1 in SDPA; scale ONLY the content part by sqrt(head_dim).
        q=self._heads(self.q_proj[index](h))/math.sqrt(self.head_dim)
        k=self._heads(self.k_proj[index](h))
        v=self._heads(self.v_proj[index](h))
        qs,ks=[q],[k]
        if self.use_time_bias:
            times=p.unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            qs.append(times*self.time_strength[index][None,:,None,None]);ks.append(times)
        if self.use_variable_bias:
            u=self.adj_U(ids).reshape(*ids.shape,self.num_heads,-1).transpose(1,2)
            w=self.adj_V(ids).reshape(*ids.shape,self.num_heads,-1).transpose(1,2)
            qs.append(u*self.variable_strength[None,:,None,None]);ks.append(w)
        q,k=torch.cat(qs,-1),torch.cat(ks,-1)
        # Equal Q/K/V feature widths permit fused SDPA kernels where available.
        width=math.ceil(q.shape[-1]/8)*8
        q,k,v=(F.pad(x,(0,width-x.shape[-1])) for x in (q,k,v))
        return q,k,v

    def _attend(self,q,k,v,degree):
        dtype=q.dtype
        if q.device.type=='cuda' and self.attention_dtype!='float32':
            name=self.attention_dtype
            if name=='auto':
                with torch.cuda.device(q.device):
                    name='bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
            work_dtype=getattr(torch,name)
            q,k,v=(x.to(work_dtype) for x in (q,k,v))
        # Prefixes are based on physical timestamps, not array positions.
        allowed = torch.arange(k.shape[-2], device=k.device) < degree.unsqueeze(-1)
        if degree.ndim == 2:
            allowed = allowed.unsqueeze(1)
        probability=self.attention_dropout if self.training else 0.
        if self.backend=='sdpa':
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,
                                              dropout_p=probability,scale=1.0)
        else:
            scores=(q@k.transpose(-1,-2)).masked_fill(~allowed,float('-inf'))
            scores = torch.where(allowed.any(-1, keepdim=True), scores, 0.)
            weights=F.dropout(torch.where(allowed, scores.softmax(-1), 0.),p=probability,training=self.training)
            out=weights@v
        return out[...,:self.head_dim].to(dtype)

    def forward(self,h,p,var_idx,masks,valid,timestamp=None,node_type=None,
                receiver_chunk_size=None,sparse_plan=None):
        plan=sparse_plan if sparse_plan is not None else build_sparse_relations(timestamp,var_idx,node_type,valid)
        if isinstance(plan, PackedAttentionPlan):
            return self._forward_packed(h, var_idx, valid, timestamp, node_type, plan)
        normalized=self.norm_attention(h)
        p=time_features(self.time_embedding,timestamp,valid) if self.use_time_bias else None
        chunk=receiver_chunk_size or 512
        message=torch.zeros_like(h)
        for index in range(len(self.q_proj)):
            q,k,v=self._qkv(normalized,p,var_idx,index)
            values=[torch.zeros_like(row) for row in h]
            active=torch.zeros_like(valid)
            relation = self.relation_ids[index]
            for group in plan:
                if group.relation!=relation:continue
                b,src,dst=group.batch,group.sources,group.receivers
                key,value=k[b,:,src],v[b,:,src]
                for start in range(0,len(dst),chunk):
                    nodes=dst[start:start+chunk];degrees=group.degrees[start:start+chunk]
                    def attend(q,k,v,degree=degrees):return self._attend(q,k,v,degree)
                    inputs=(q[b,:,nodes],key,value)
                    out=checkpoint(attend,*inputs,use_reentrant=False) if self.checkpoint_attention and self.training and torch.is_grad_enabled() else attend(*inputs)
                    out=out.transpose(0,1).reshape(len(nodes),-1)
                    values[b]=values[b].index_add(0,nodes,out)
                active[b,dst]=True
            # Empty attention rows have zero messages, including output bias.
            zero=(q.sum()+k.sum()+v.sum())*0
            projected=self.out_proj[index](torch.stack(values)+zero)
            message=message+torch.where(active.unsqueeze(-1),projected,0)
        out=h+self.dropout(message)
        out=out+self.dropout(self.ffn(self.norm_ffn(out)))
        return self._select_receivers(out, h, valid, node_type)

    def _select_receivers(self, out, h, valid, node_type):
        exogenous = (node_type == 0) | (node_type == 2)
        if self.cross:
            out = torch.where((valid & ~exogenous).unsqueeze(-1), out, h)
        if self.within_group:
            selected = (exogenous & (3 in self.relation_ids)) | (~exogenous & (4 in self.relation_ids))
            out = torch.where((valid & selected).unsqueeze(-1), out, h)
        return torch.where(valid.unsqueeze(-1), out, 0)


    def _forward_packed(self, h, var_idx, valid, timestamp, node_type, plan):
        normalized = self.norm_attention(h)
        p = time_features(self.time_embedding, timestamp, valid) if self.use_time_bias else None
        B, L, D = h.shape
        message = torch.zeros_like(h)
        for index in range(len(self.q_proj)):
            q, k, v = self._qkv(normalized, p, var_idx, index)
            # Flat observation indices keep gather/scatter batched across records.
            q, k, v = (x.transpose(1, 2).reshape(B * L, self.num_heads, -1) for x in (q, k, v))
            value = h.new_zeros(B * L, D)
            active = torch.zeros(B * L, dtype=torch.bool, device=h.device)
            relation = self.relation_ids[index]
            for bucket in plan.buckets:
                if bucket.relation != relation:
                    continue
                qq = q[bucket.receiver].permute(0, 2, 1, 3)
                kk = k[bucket.source].permute(0, 2, 1, 3)
                vv = v[bucket.source].permute(0, 2, 1, 3)
                def attend(q, k, v, degree=bucket.degrees):
                    return self._attend(q, k, v, degree)
                if self.checkpoint_attention and self.training and torch.is_grad_enabled():
                    out = checkpoint(attend, qq, kk, vv, use_reentrant=False)
                else:
                    out = attend(qq, kk, vv)
                out = out.transpose(1, 2).reshape(-1, D)[bucket.positions]
                value = value.index_add(0, bucket.destinations, out)
                active[bucket.destinations] = True
            zero = (q.sum() + k.sum() + v.sum()) * 0
            projected = self.out_proj[index](value.reshape(B, L, D) + zero)
            message = message + torch.where(active.reshape(B, L, 1), projected, 0)
        out = h + self.dropout(message)
        out = out + self.dropout(self.ffn(self.norm_ffn(out)))
        return self._select_receivers(out, h, valid, node_type)


class ExEnGraphTransformer(ExEn):
    """Time, optional within-role variable, and X-to-Y attention stages."""
    def __init__(self,*args,num_heads=2,ffn_multiplier=2,attention_backend='sdpa',
                 attention_dtype='float32',attention_dropout=0.,use_time_bias=True,
                 use_variable_bias=True,time_bias_init=1.,variable_bias_init=.01,
                 time_embedding_type='rbf',receiver_chunk_size=512,
                 gcn_type='attention',gcn_depth=1,relation_backend='sparse_csr',checkpoint_attention=True,
                 n_variable_layer=0, within_exogenous=True, within_endogenous=True, **kwargs):
        if relation_backend!='sparse_csr':
            raise ValueError('Use grouped source plan (relation_backend=sparse_csr); attention_backend selects the kernel')
        if not isinstance(n_variable_layer, int) or isinstance(n_variable_layer, bool) or n_variable_layer < 0:
            raise ValueError('n_variable_layer must be a nonnegative integer')
        if not isinstance(within_exogenous, bool) or not isinstance(within_endogenous, bool):
            raise ValueError('Within-group switches must be booleans')
        self._attention_options=dict(num_heads=num_heads,ffn_multiplier=ffn_multiplier,
            attention_backend=attention_backend,attention_dtype=attention_dtype,attention_dropout=attention_dropout,
            use_time_bias=use_time_bias,use_variable_bias=use_variable_bias,
            time_bias_init=time_bias_init,variable_bias_init=variable_bias_init,time_embedding_type=time_embedding_type,checkpoint_attention=checkpoint_attention)
        super().__init__(*args,gcn_type=gcn_type,gcn_depth=gcn_depth,time_embedding_type=time_embedding_type,
                         receiver_chunk_size=receiver_chunk_size,relation_backend=relation_backend,**kwargs)
        self.within_relations = tuple(r for r, enabled in ((3, within_exogenous), (4, within_endogenous)) if enabled)
        self.variable_layers = nn.ModuleList([
            GraphAttentionLayer(*self._relation_layer_args, relations=self.within_relations, **self._attention_options)
            for _ in range(n_variable_layer if self.within_relations else 0)])

    def _validate_depths(self, n_layer, n_cross_layer):
        for value in (n_layer, n_cross_layer):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError('Graph Transformer stage depths must be nonnegative integers')

    def _apply_variable_layers(self, h, p, var_idx, valid, timestamp, node_type, plan):
        if not self.variable_layers:
            return h
        if isinstance(plan, PackedAttentionPlan):
            if plan.within_relations != self.within_relations:
                # Support callers that built a legacy temporal-only packed plan.
                metadata = [x.detach().cpu() for x in (timestamp, var_idx, node_type, valid)]
                groups = build_within_variable_relations(*metadata, self.within_relations)
                plan = pack_attention_plan(*metadata, groups=groups,
                    within_relations=self.within_relations).to(h.device)
        else:
            plan = build_within_variable_relations(timestamp, var_idx, node_type, valid, self.within_relations)
        for layer in self.variable_layers:
            h = layer(h, p, var_idx, None, valid, timestamp, node_type,
                      self.receiver_chunk_size, sparse_plan=plan)
        return h

    def _validate_operator(self,gcn_type,gcn_depth):
        if gcn_type!='attention' or gcn_depth!=1:
            raise ValueError('ExEnGraphTransformer uses attention, not a Chebyshev/GCN operator')

    def _make_relation_layer(self,*args,cross=False):
        return GraphAttentionLayer(*args,cross=cross,**self._attention_options)
