import json
import sys
import transformers
from transformers import LlamaTokenizer
import torch

# fn, tok_path = sys.argv[1], sys.argv[2]
fn = sys.argv[1]
out_fn = sys.argv[2]

# tokenizer = LlamaTokenizer.from_pretrained(
#     tok_path,
#     use_fast=False,
#     padding_side="left",
# )

with open(fn, 'r') as f:
    d = json.load(f)

for key in d:
    scores = d[key]['pred_answers_scores']

    # convert to pt
    t = torch.tensor(scores)
    torch.save(t, f"{out_fn}.{key}.pt")
