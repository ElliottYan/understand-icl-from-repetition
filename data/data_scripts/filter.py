import json
import random
import os
import sys
from transformers import AutoTokenizer, LlamaTokenizer

assert len(sys.argv) >= 3
input_file_path = sys.argv[1]
tok_name = sys.argv[2]

if tok_name.startswith('llama'):
    tokenizer = LlamaTokenizer.from_pretrained(tok_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

output_file_path = input_file_path + '.{}.filter_1k'.format(tok_name)

# Read the JSON file line by line
encoded_lines = []
with open(input_file_path, "r") as input_file:
    for line in input_file:
#         data = json.loads(line)
        text = line.strip()
        encoded = tokenizer.encode(text, return_tensors="pt")
        if len(encoded[0]) < 10:
            encoded_lines.append(text)

# Randomly sample 1000 lines
sampled_encoded_lines = random.sample(encoded_lines, 1000)

# Convert the sampled lines back to text
sampled_lines = [txt for txt in sampled_encoded_lines]

# Write the sampled lines to a new JSON file
with open(output_file_path, "w") as output_file:
    for line in sampled_lines:
        output_file.write(line + '\n')