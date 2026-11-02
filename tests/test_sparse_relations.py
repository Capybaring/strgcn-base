import pytest
import torch
from models import ExEn
from models.STRGCN.ExEn import relation_masks
from models.STRGCN.SparseRelations import build_sparse_relations
from test_exen import graph


@pytest.mark.parametrize('chunk',[1,3,128])
@pytest.mark.parametrize('embedding',['rbf','rope','transformer'])
def test_sparse_outputs_and_all_parameter_gradients_match_dense(chunk,embedding):
    torch.manual_seed(5)
    args=dict(num_vars=2,exogenous_var_indices=[0],hid_dim=8,n_layer=2,n_cross_layer=2,
              time_embedding_type=embedding)
    dense=ExEn(**args)
    sparse=ExEn(**args,relation_backend='sparse_csr',receiver_chunk_size=chunk)
    sparse.load_state_dict(dense.state_dict())
    b=graph()
    b={k:v.repeat(2,1) for k,v in b.items()}
    b['batch_pad_mask'][1,0]=0
    b['batch_value'][0,4:6]=torch.nan
    p=dense(**b);q=sparse(**b)
    torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
    p[:,4:6].square().sum().backward();q[:,4:6].square().sum().backward()
    for (name,a),(other,c) in zip(dense.named_parameters(),sparse.named_parameters()):
        assert name==other
        if a.grad is None:
            assert c.grad is None or (c.grad==0).all(),name
        else:
            assert c.grad is not None,name
            torch.testing.assert_close(a.grad,c.grad,rtol=3e-4,atol=3e-5,msg=name)
    dense.eval();sparse.eval()
    with torch.no_grad():torch.testing.assert_close(dense(**b),sparse(**b),rtol=2e-5,atol=2e-6)


def test_sparse_plan_equals_masks_with_unordered_and_tied_times():
    b=graph()
    b['batch_timestamp'][0,2]=0  # equal-time neighbors are included
    order=torch.tensor([6,3,5,0,2,4,1])
    b={k:v[:,order] for k,v in b.items()}
    valid=b['batch_pad_mask'].bool()
    groups=build_sparse_relations(b['batch_timestamp'],b['batch_var_idx'],b['batch_node_type'],valid)
    expected=relation_masks(b['batch_timestamp'],b['batch_var_idx'],b['batch_node_type'],valid)
    actual=[torch.zeros_like(m) for m in expected]
    for g in groups:
        for target,degree in zip(g.receivers,g.degrees):
            actual[g.relation][g.batch,target,g.sources[:int(degree)]]=True
    for a,e in zip(actual,expected):assert torch.equal(a,e)


def test_sparse_query_gradient_and_later_exogenous_causality():
    torch.manual_seed(3)
    m=ExEn(num_vars=2,exogenous_var_indices=[0],hid_dim=8,n_layer=3,
           relation_backend='sparse_csr',receiver_chunk_size=2).eval()
    b=graph();b['batch_value'].requires_grad_()
    before=m(**b);before[:,4:6].sum().backward()
    assert (b['batch_value'].grad[:,4:6]==0).all()
    with torch.no_grad():
        b['batch_value'][0,6]+=100
        after=m(**b)
    torch.testing.assert_close(before[:,:5],after[:,:5])
    assert not torch.allclose(before[:,5],after[:,5])


def test_sparse_empty_neighbors_and_padding_are_finite():
    m=ExEn(num_vars=2,exogenous_var_indices=[0],hid_dim=8,relation_backend='sparse_csr')
    b={k:v[:,4:5] for k,v in graph().items()}
    out=m(**b);assert torch.isfinite(out).all()
    out.sum().backward()
    b['batch_pad_mask'].zero_()
    assert (m(**b)==0).all()


def test_sparse_absent_relation_has_same_zero_gradients():
    torch.manual_seed(4)
    args=dict(num_vars=2,exogenous_var_indices=[0],hid_dim=8)
    dense=ExEn(**args)
    sparse=ExEn(**args,relation_backend='sparse_csr')
    sparse.load_state_dict(dense.state_dict())
    b={k:v[:,4:5] for k,v in graph().items()}  # a query with no edges at all
    dense(**b).sum().backward();sparse(**b).sum().backward()
    for (name,a),(other,c) in zip(dense.named_parameters(),sparse.named_parameters()):
        assert name==other
        if a.grad is not None:
            assert c.grad is not None,name
            torch.testing.assert_close(a.grad,c.grad,atol=1e-6,rtol=1e-5,msg=name)
