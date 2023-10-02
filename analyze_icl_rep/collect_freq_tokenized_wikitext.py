import tqdm
import sys
import jsonlines

from analyze_icl_rep.utils import *
from transformers import LlamaTokenizer

from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
 
# we use the stopwords found here: https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt

model_path = sys.argv[1]
# output_path = sys.argv[2]

input_file = 'data/wikitext-103-raw/wiki.train.raw'
# input_file = 'data/wikitext-103-raw/wiki.valid.raw'

if 'llama' in model_path:
    ms = 'llama'
elif 'opt' in model_path:
    ms = 'opt'
elif 'gpt' in model_path: ms = 'gpt'
else:
    raise
cache_file = input_file + '.{}.idx'.format(ms)

if not os.path.exists(cache_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()
    # split docs based on \n
    docs = [line.strip() for line in lines if line.strip() != ""]

    if 'llama' in model_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

    tokenized = []
    for doc in tqdm.tqdm(docs):
        ids = tokenizer.encode(doc)
        if ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        str_ids = [str(item) for item in ids]
        tokenized.append(" ".join(str_ids))

    with open(cache_file, 'w', encoding='utf8') as f:
        for line in tokenized:
            f.write(line + '\n')
else:
    # read cache
    with open(cache_file, 'r', encoding='utf8') as f:
        # for line in tokenized:
        lines = f.readlines()
    tokenized = [line.strip() for line in lines]

print("Computing Counter Vectorizer.")
vec_path = cache_file + '.vec.pkl'
if os.path.exists(vec_path):
    print("Reading from cache.")
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    # fit tfidf of 2-gram

    # Creating a simple corpus
    # corpus = [
    #     'This is the first document.',
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'Is this the first document?',
    # ]

    vectorizer = CountVectorizer(ngram_range=(2, 2))

    # Fit and transform the corpus
    X = vectorizer.fit_transform(tokenized)

    # To see the features (in this case, bigrams)
    print(vectorizer.get_feature_names_out())

    with open(vec_path, 'wb') as fin:
        pickle.dump(vectorizer, fin)

print("Cuting into bins.")
nbins = 5
bin_path = input_file + f'.{ms}.nb-{nbins}.csv'
if os.path.exists(bin_path):
    print("Already computed.")
else:
    df = pd.DataFrame([vectorizer.vocabulary_])
    bin_df = pd.cut(df.iloc[0], bins=nbins, labels=False)
    # save to csv
    bin_df.to_csv(bin_path, header=False, index=False)