import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda:0")
DTYPE = torch.float16

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=DTYPE, device_map="auto")
model.half()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

handles = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
print(f"Linear layers: {len(handles)}")

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
texts = [t["text"] for t in ds if t["text"].strip()]
sample_texts = texts[:4]
enc = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)

acts = {n: [] for n, _ in handles}
hooks = []
def mk(name):
    def hook(mod, inp, out):
        acts[name].append(inp[0].detach().cpu())
    return hook
for n, m in handles:
    hooks.append(m.register_forward_hook(mk(n)))

input_ids = enc["input_ids"].to(device)
attn_mask = enc["attention_mask"].to(device)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids, attention_mask=attn_mask)
t1 = time.time()
print(f"One forward: {t1-t0:.2f}s")

for h in hooks:
    h.remove()
