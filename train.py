import torch
torch.manual_seed(1337)

## Hyperparameters for training
max_iters = 1000
eval_interval = 500
learning_rate= 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
batch_size=4
block_size = 8

def get_batch(split):
  data = train_data if split=="train" else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y


optimizer = torch.optim.Adam(m.parameters(), lr= learning_rate)
for steps in range(max_iters):
  xb,yb = get_batch('train')
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  print(f"Train loss at epoch {steps} = {loss.item()}")
  if steps%eval_interval == 0:
    xv,yv = get_batch("val")
    logits, loss = m(xv,yv)
    print(f"val loss = {loss.item()}" )