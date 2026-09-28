"""Working classroom baseline: a small GPT trained from random initialization."""
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Parameter-matched gated feed-forward network."""
    def __init__(self, width):
        super().__init__()
        hidden = 8 * width // 3
        self.gate_value = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


def make_norm(width, kind):
    if kind == 'layernorm':
        return nn.LayerNorm(width)
    if kind == 'rmsnorm':
        return nn.RMSNorm(width, eps=1e-5)
    raise ValueError(f'Unknown norm: {kind}')


class Block(nn.Module):
    def __init__(self, width=128, heads=4, norm='layernorm', mlp='gelu'):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = make_norm(width, norm), make_norm(width, norm)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        if mlp == 'gelu':
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
            )
        elif mlp == 'swiglu':
            self.mlp = SwiGLU(width)
        else:
            raise ValueError(f'Unknown MLP: {mlp}')

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        norm = config.get('norm', 'layernorm')
        mlp = config.get('mlp', 'gelu')
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], norm, mlp) for _ in range(config['depth'])
        ])
        self.norm = make_norm(width, norm)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        if config.get('tie_embeddings', True):
            self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Override this for a strictly causal, within-window memory mechanism.
        A prediction at position t can use ids[:, :t+1] and nothing later.
        Reset all temporary state on every call; each evaluation window starts fresh.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Keep the classroom baseline runnable with --implementation model."""
    return GPT(config)
