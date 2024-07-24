import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

## Hyperparameters
batch_size = 64
block_size = 256 # what is the maximum context length for the predictions?
max_iters = 1000
eval_interval = 500
learning_rate= 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer =6
n_head = 6
dropout = 0.2

# ---------------------------------------------------

class Head(nn.Module):
  """one head of self-attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)
    v = self.value(x) # (B,T,C)

    # compute self-attention or affinities among the key and query

    wei = q @ k.transpose(-2,-1)  #(B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) #(B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out =  torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation"""

  def __init__(self, n_embd, n_head):
    # n_embd : embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # x + self.sa is the residual skip connections
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, idx, targets=None):
    B,T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device= device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is the (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits,loss = self(idx_cond)
      # forcus only on the last time step
      logits = logits[:, -1, :]
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      # applend sampled index to the running sequene
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


### Initial loss will be -ln(1/65) as we have 65 possible vocabulary elements
