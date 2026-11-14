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
                 time_bias_init=1., variable_bias_init=.01, checkpoint_attention=True, relations=None,
                 window_conditioning=False):
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
        self.window_query = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(count)]) if window_conditioning else None
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

    def _qkv(self,h,p,ids,index,window_state=None):
        # scale=1 in SDPA; scale ONLY the content part by sqrt(head_dim).
        content_q = self.q_proj[index](h)
        if window_state is not None:
            content_q = content_q + self.window_query[index](window_state).unsqueeze(1)
        q=self._heads(content_q)/math.sqrt(self.head_dim)
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
                receiver_chunk_size=None,sparse_plan=None,receiver_mask=None,window_state=None,
                window_layout=None,window_cache=None,window_time=None):
        if window_layout is not None:
            return self._forward_window(h, var_idx, valid, node_type, window_layout,
                window_cache, window_time, window_state)
        plan=sparse_plan if sparse_plan is not None else build_sparse_relations(timestamp,var_idx,node_type,valid)
        if isinstance(plan, PackedAttentionPlan):
            return self._forward_packed(h, var_idx, valid, timestamp, node_type, plan, receiver_mask, window_state)
        normalized=self.norm_attention(h)
        p=time_features(self.time_embedding,timestamp,valid) if self.use_time_bias else None
        chunk=receiver_chunk_size or 512
        message=torch.zeros_like(h)
        for index in range(len(self.q_proj)):
            q,k,v=self._qkv(normalized,p,var_idx,index,window_state)
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
        return self._select_receivers(out, h, valid, node_type, receiver_mask)

    def _select_receivers(self, out, h, valid, node_type, receiver_mask=None):
        exogenous = (node_type == 0) | (node_type == 2)
        if self.cross:
            out = torch.where((valid & ~exogenous).unsqueeze(-1), out, h)
        if self.within_group:
            selected = (exogenous & (3 in self.relation_ids)) | (~exogenous & (4 in self.relation_ids))
            out = torch.where((valid & selected).unsqueeze(-1), out, h)
        if receiver_mask is not None:
            out = torch.where(receiver_mask.unsqueeze(-1), out, h)
        return torch.where(valid.unsqueeze(-1), out, 0)


    def _forward_packed(self, h, var_idx, valid, timestamp, node_type, plan, receiver_mask=None, window_state=None):
        normalized = self.norm_attention(h)
        p = time_features(self.time_embedding, timestamp, valid) if self.use_time_bias else None
        B, L, D = h.shape
        message = torch.zeros_like(h)
        for index in range(len(self.q_proj)):
            q, k, v = self._qkv(normalized, p, var_idx, index, window_state)
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
        return self._select_receivers(out, h, valid, node_type, receiver_mask)


    def _window_width(self):
        width = self.head_dim
        if self.use_time_bias:
            width += self.time_embedding.hid_dim
        if self.use_variable_bias:
            width += self.adj_U.embedding_dim // self.num_heads
        return math.ceil(width / 8) * 8

    def _window_query(self, normalized, times, ids, index, state, batches):
        n = normalized.shape[0]
        content = self.q_proj[index](normalized)
        content = content + self.window_query[index](state)[batches]
        parts = [content.reshape(n, self.num_heads, self.head_dim) / math.sqrt(self.head_dim)]
        if self.use_time_bias:
            parts.append(times[:, None, :].expand(-1, self.num_heads, -1) *
                         self.time_strength[index][None, :, None])
        if self.use_variable_bias:
            parts.append(self.adj_U(ids).reshape(n, self.num_heads, self.adj_U.embedding_dim // self.num_heads) *
                         self.variable_strength[None, :, None])
        q = torch.cat(parts, -1)
        return F.pad(q, (0, self._window_width() - q.shape[-1]))

    def _window_kv(self, normalized, times, ids, index):
        n = normalized.shape[0]
        key = self.k_proj[index](normalized).reshape(n, self.num_heads, self.head_dim)
        value = self.v_proj[index](normalized).reshape(n, self.num_heads, self.head_dim)
        parts = [key]
        if self.use_time_bias:
            parts.append(times[:, None, :].expand(-1, self.num_heads, -1))
        if self.use_variable_bias:
            parts.append(self.adj_V(ids).reshape(n, self.num_heads,
                         self.adj_V.embedding_dim // self.num_heads))
        key = torch.cat(parts, -1)
        width = self._window_width()
        return F.pad(key, (0, width - key.shape[-1])), F.pad(value, (0, width - value.shape[-1]))

    def _cache_final_observations(self, h, times, ids):
        # Keys/values depend on finalized observations and this layer's parameters,
        # not the next window state (which affects Q only). Keep autograd attached.
        normalized = self.norm_attention(h)
        return [self._window_kv(normalized, times, ids, r) for r in range(len(self.q_proj))]

    def _forward_window(self, h, ids, valid, types, layout, cache, times, state):
        """Only current receivers and current known sources execute node networks."""
        current = h[0]
        normalized = self.norm_attention(current)
        known_h, known_ids = normalized[layout.known], ids[0, layout.known]
        known_time = times[layout.known] if times is not None else None
        n, dim = current.shape
        message = torch.zeros_like(current)
        for index, relation in enumerate(self.relation_ids):
            q = self._window_query(normalized, times, ids[0], index, state, layout.batch)
            local_k, local_v = self._window_kv(known_h, known_time, known_ids, index)
            chunks = cache[index]
            k = torch.cat([entry[0] for entry in chunks] + [local_k], 0)
            v = torch.cat([entry[1] for entry in chunks] + [local_v], 0)
            value = current.new_zeros(n, dim)
            active = torch.zeros(n, dtype=torch.bool, device=h.device)
            for bucket in layout.plan.buckets:
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
                out = out.transpose(1, 2).reshape(-1, dim)[bucket.positions]
                value = value.index_add(0, bucket.destinations, out)
                active[bucket.destinations] = True
            zero = (q.sum() + k.sum() + v.sum()) * 0
            projected = self.out_proj[index](value + zero)
            message = message + torch.where(active[:, None], projected, 0)
        out = current + self.dropout(message)
        out = out + self.dropout(self.ffn(self.norm_ffn(out)))
        return self._select_receivers(out[None], h, valid, types)


class ExEnGraphTransformer(ExEn):
    """Time, optional within-role variable, and X-to-Y attention stages."""
    def __init__(self,*args,num_heads=2,ffn_multiplier=2,attention_backend='sdpa',
                 attention_dtype='float32',attention_dropout=0.,use_time_bias=True,
                 use_variable_bias=True,time_bias_init=1.,variable_bias_init=.01,
                 time_embedding_type='rbf',receiver_chunk_size=512,
                 gcn_type='attention',gcn_depth=1,relation_backend='sparse_csr',checkpoint_attention=True,
                 n_variable_layer=0, within_exogenous=True, within_endogenous=True, rolling_window_size=None, **kwargs):
        if relation_backend!='sparse_csr':
            raise ValueError('Use grouped source plan (relation_backend=sparse_csr); attention_backend selects the kernel')
        if not isinstance(n_variable_layer, int) or isinstance(n_variable_layer, bool) or n_variable_layer < 0:
            raise ValueError('n_variable_layer must be a nonnegative integer')
        if not isinstance(within_exogenous, bool) or not isinstance(within_endogenous, bool):
            raise ValueError('Within-group switches must be booleans')
        if rolling_window_size is not None and (isinstance(rolling_window_size, bool) or
                not isinstance(rolling_window_size, (int, float)) or not math.isfinite(rolling_window_size) or rolling_window_size <= 0):
            raise ValueError('rolling_window_size must be a positive finite number or null')
        self.rolling_window_size = rolling_window_size
        self._attention_options=dict(num_heads=num_heads,ffn_multiplier=ffn_multiplier,
            attention_backend=attention_backend,attention_dtype=attention_dtype,attention_dropout=attention_dropout,
            use_time_bias=use_time_bias,use_variable_bias=use_variable_bias,
            time_bias_init=time_bias_init,variable_bias_init=variable_bias_init,time_embedding_type=time_embedding_type,checkpoint_attention=checkpoint_attention,
            window_conditioning=rolling_window_size is not None)
        super().__init__(*args,gcn_type=gcn_type,gcn_depth=gcn_depth,time_embedding_type=time_embedding_type,
                         receiver_chunk_size=receiver_chunk_size,relation_backend=relation_backend,**kwargs)
        self.within_relations = tuple(r for r, enabled in ((3, within_exogenous), (4, within_endogenous)) if enabled)
        self.variable_layers = nn.ModuleList([
            GraphAttentionLayer(*self._relation_layer_args, relations=self.within_relations, **self._attention_options)
            for _ in range(n_variable_layer if self.within_relations else 0)])
        if rolling_window_size is not None:
            from .RollingWindows import WindowPatternMemory
            self.window_memory = WindowPatternMemory(self.encoder.out_features, self.num_vars, num_heads)

    def _run_graph_layers(self, h, p, var_idx, valid, tt, types, sparse_plan, masks):
        if self.rolling_window_size is None:
            return super()._run_graph_layers(h, p, var_idx, valid, tt, types, sparse_plan, masks)
        from .RollingWindows import pack_rolling_attention_plan
        if not isinstance(sparse_plan, PackedAttentionPlan) or sparse_plan.windows is None:
            # Direct model calls may omit dataset metadata. Use actual observed bounds;
            # production datasets supply exact task/forecast bounds in CPU collation.
            meta = [x.detach().cpu() for x in (tt, var_idx, types, valid)]
            sparse_plan = pack_rolling_attention_plan(*meta, self.rolling_window_size,
                within_relations=self.within_relations).to(h.device)
        return self._run_rolling_windows(h, p, var_idx, valid, tt, types, sparse_plan)

    def _run_rolling_reference(self, h, p, var_idx, valid, tt, types, sparse_plan):
        """Original full-tensor execution retained as a numerical test oracle."""
        tokens, token_valid = [], []
        for window in sparse_plan.windows:
            state = self.window_memory.predict(tokens, token_valid, window.bounds)
            stages = []
            for layers in (self.temporal_layers, self.variable_layers, self.cross_layers):
                for layer in layers:
                    h = layer(h, p, var_idx, None, valid, tt, types,
                        self.receiver_chunk_size, sparse_plan=window.plan,
                        receiver_mask=window.receivers, window_state=state)
                stages.append(h)
            # Past nodes remain frozen in h. Only known current observations become
            # sources or summary inputs; query representations NEVER leave a window.
            known = window.receivers & (types != 3)
            token = self.window_memory.summarize(stages, tt, var_idx, types, known, window.bounds)
            tokens.append(token)
            token_valid.append(window.present)
        return h


    def _run_rolling_windows(self, h, p, var_idx, valid, tt, types, sparse_plan):
        layers = [*self.temporal_layers, *self.variable_layers, *self.cross_layers]
        flat_h = h.reshape(-1, h.shape[-1])
        flat_t, flat_ids, flat_types = (x.reshape(-1) for x in (tt, var_idx, types))
        # Timestamp-only features are invariant across windows; compute once/layer.
        time_cache = {layer: time_features(layer.time_embedding, tt, valid).reshape(-1, h.shape[-1])
                      if layer.use_time_bias else None for layer in layers}
        kv_cache = {layer: [[] for _ in layer.relation_ids] for layer in layers}
        tokens, token_valid, outputs, positions = [], [], [], []
        for number, window in enumerate(sparse_plan.windows):
            layout = window.layout
            index = layout.current
            ids, roles, times = flat_ids[index][None], flat_types[index][None], flat_t[index][None]
            current = flat_h[index][None]
            current_valid = torch.ones_like(ids, dtype=torch.bool)
            state = self.window_memory.predict(tokens, token_valid, window.bounds)
            stages = []
            for block in (self.temporal_layers, self.variable_layers, self.cross_layers):
                for layer in block:
                    features = time_cache[layer]
                    features = features[index] if features is not None else None
                    current = layer(current, None, ids, None, current_valid, times, roles,
                        sparse_plan=layout.plan, receiver_mask=current_valid, window_state=state,
                        window_layout=layout, window_cache=kv_cache[layer], window_time=features)
                stages.append(current)
            outputs.append(current[0]); positions.append(index)
            # Summaries are padded only to this window's observed count, never L.
            selected = layout.summary
            token = self.window_memory.summarize([stage[0, selected] for stage in stages],
                times[0, selected], ids[0, selected], roles[0, selected],
                layout.summary_valid, window.bounds)
            tokens.append(token); token_valid.append(window.present)
            if number + 1 < len(sparse_plan.windows):
                known = layout.known
                for layer in layers:
                    features = time_cache[layer]
                    features = features[index[known]] if features is not None else None
                    entries = layer._cache_final_observations(current[0, known], features, ids[0, known])
                    for history, entry in zip(kv_cache[layer], entries):
                        history.append(entry)
        if outputs:
            flat_h = flat_h.index_copy(0, torch.cat(positions), torch.cat(outputs))
        return flat_h.reshape_as(h)


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
