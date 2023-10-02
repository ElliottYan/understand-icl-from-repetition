import tqdm
import sys
import jsonlines

from utils import *
from transformers import LlamaTokenizer
 
# we use the stopwords found here: https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt

model_path = sys.argv[1]
output_path = sys.argv[2]

with open('data/stopwords-en.txt', 'r') as f:
    lines = f.readlines()
    stop_words = set([w.strip() for w in lines])

if 'llama' in model_path:
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

tokenized = []
for w in tqdm.tqdm(stop_words):
    ids = tokenizer.encode([w, ], is_split_into_words=True)
    if ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    tokenized.append((w, ids))

with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(tokenized)
