from torch import nn
import einops as E


class LogSoftmaxClass(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=dim)

    def forward(self, x):
        s, m, k = x.shape[1:4]
        x = E.rearrange(x, 'b s m k c h w -> (b s m) k c h w')
        x = self.softmax(x)
        x = E.rearrange(x, '(b s m) k c h w -> b s m k c h w', s=s, m=m, k=k)
        return x
    

class SoftmaxClass(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        s, m, k = x.shape[1:4]
        x = E.rearrange(x, 'b s m k c h w -> (b s m) k c h w')
        x = self.softmax(x)
        x = E.rearrange(x, '(b s m) k c h w -> b s m k c h w', s=s, m=m, k=k)
        return x