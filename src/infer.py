from transformer import BigramLanguageModel



m = BigramLanguageModel()
out, loss = m(xb,yb)
print(out.shape)
print(loss)

id = torch.zeros((1,1), dtype= torch.long)
print(decode(m.generate(id,max_new_tokens=100)[0].tolist()))