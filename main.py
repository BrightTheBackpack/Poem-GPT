import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import os
import gradio as gr


block_size = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
eval_iters = 50
batch_size = 32
max_iters = 20000000
learning_rate = 1e-4
n_embed = 384
n_layer = 12
n_head = 4
dropout = 0.2


with open('all_poems.txt', 'r', encoding="utf8") as file:
    text = file.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# BigramLanguageModel
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed,head_size, bias=False)
    self.query = nn.Linear(n_embed,head_size, bias=False)
    self.value = nn.Linear(n_embed,head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)))
  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for  _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

class FeedFoward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4*  n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),

    )
  def forward(self, x):
    return self.net(x)
class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)

    self.ffwd = FeedFoward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
  def forward(self,x):
    x = x + self.sa(self.ln1(x))
    x = x+   self.ffwd(self.ln2(x))
    return x
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

    self.ln_f = nn.LayerNorm(n_embed)
    self.sa_head = MultiHeadAttention(4, n_embed//4)#MultiHeadAttention(4, n_embed//4)
    self.ffwd = FeedFoward(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
  def forward(self, idx, targets=None):
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx) #B,T,C
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))[None,:,:] #1,T,C
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    return logits, loss
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

# Checkpoint handling
def save_checkpoint(iteration, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': iteration,
        'loss': loss,
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, 'checkpoint.pth')
    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint():
    if not os.path.exists('checkpoint.pth'):
        print("No checkpoint found. Starting training from the beginning.")
        return 0

    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict']) 
    print("Model loaded")
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])# uncomment for training
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # iteration = checkpoint['iteration']
    
    # random.setstate(checkpoint['random_state'])
    # torch.set_rng_state(checkpoint['torch_rng_state'])
    # if torch.cuda.is_available():
    #     torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    # print(f"Resuming training from iteration {iteration}")
    # return iteration

# Main training loop
def train():
    start_iter = load_checkpoint()
    last_saved_loss = None 
    
    try:
        for iter in range(start_iter, max_iters):
     
            if iter % eval_iters == 0:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # Generate some text
                start_word = random.choice(text.split())
                encoded_word = encode(start_word)
                context = torch.tensor(encoded_word, dtype=torch.long, device=device).unsqueeze(0)
                generated = model.generate(context, max_new_tokens=500)[0].tolist()
                print("Generated text:", start_word,decode(generated))

            # Get batch and calculate loss
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            last_saved_loss = loss.item()  # Update the last known loss

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

   
            if iter % 1000 == 0:
                save_checkpoint(iter, last_saved_loss)

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(iter, last_saved_loss)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if last_saved_loss is not None:
            save_checkpoint(iter, last_saved_loss)
        else:
            print("No loss value available. Saving checkpoint without loss information.")
            save_checkpoint(iter, None)

    print("Training completed.")
    torch.save(model.state_dict(), 'final_model.pth')

def predict(prompt, max_tokens):
    try:
        # Encode the text and generate predictions
        start_word = prompt
        encoded_word = encode(start_word)
        context = torch.tensor(encoded_word, dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(context, max_new_tokens=max_tokens)[0].tolist()

        # Decode the generated output
        output_text = decode(generated)
        return output_text
    except Exception as e:
        return str(e)
interface = gr.Interface(
    fn=predict,  # Function to call
    inputs=[gr.Textbox(label="Enter Text"), gr.Number(label="Enter Number")],
    outputs=gr.Textbox(label="Generated Text")  # Use gr.Textbox for text output
)

if __name__ == "__main__":
  load_checkpoint()
  interface.launch() #train() for training


