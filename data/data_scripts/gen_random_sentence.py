import random
from transformers import LlamaTokenizer, AutoTokenizer
import os
import sys
import json

assert len(sys.argv) >= 3
tok_name = sys.argv[2]
output_file_path = sys.argv[1]

assert output_file_path.endswith('.idx'), "Output path should end with .idx"

if tok_name.startswith('llama'):
    tokenizer = LlamaTokenizer.from_pretrained(tok_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

# Generate 1000 lines of random tokens with a maximum length of 20 tokens
random_idx_lists = []
idx_list = list(range(len(tokenizer)))
for _ in range(1000):
    line_length = random.randint(5, 10)

    random_idx = [random.choice(idx_list) for _ in range(line_length)]
    random_idx_lists.append(random_idx)

# Write the random lines to the output file
with open(output_file_path, "w") as output_file:
    for idxes in random_idx_lists:
        if output_file_path.endswith('jsonl'):
            output_file.write(json.dumps(idxes)+'\n')
        else:
            output_file.write(f'{" ".join(map(str, idxes))}\n')