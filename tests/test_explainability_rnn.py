import torch
from torch import nn
from hicopd_unimetricnet.explainability import _rnn_backward_ready


def test_rnn_backward_context_is_deterministic_and_restores_eval():
    gru=nn.GRU(3,4,batch_first=True,dropout=0.0)
    gru.eval()
    x=torch.randn(2,5,3,requires_grad=True)
    assert gru.training is False
    with _rnn_backward_ready(gru):
        assert gru.training is True
        y,_=gru(x)
        y.sum().backward()
    assert gru.training is False
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
