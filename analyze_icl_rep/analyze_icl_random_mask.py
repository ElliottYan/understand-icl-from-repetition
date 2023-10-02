import pickle
import os
import math
import torch
import gc
import tqdm

from analyze_icl_rep.utils import *

@torch.no_grad()
def single_sentence_eval_icl(prefix_sentence, base_sentence, model, tokenizer, iterate_num=10, random_percentage=0.1):
    device = 'cuda'
    # [prefix + iterative_num * icl_examples]
    # each icl_example differs from base sentence within a certain range
    # currently only deals with one sentences each time.
    if isinstance(base_sentence, list):
        assert len(base_sentence) == 1
        base_sentence = base_sentence[0]
        prefix_sentence = prefix_sentence[0] + " "
    base_tokens = tokenizer.encode(base_sentence, truncation=False, return_tensors="pt").to(device)
    prefix_tokens = tokenizer.encode(prefix_sentence, truncation=False, return_tensors="pt").to(device)
    prefix_len = prefix_tokens.shape[1]
   
    # Remove possible bos
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
    # NOTE: if the tokenizer adds eos, should deal with that.

    # generate random mask with exactly, say 10% of the tokens are masked.
    assert 0 <= random_percentage < 1, "Mask percentage must be between 0 and 1."

    # Calculate the number of elements that need to be masked
    total_elements = base_tokens.numel()
    num_masked_elements = math.floor(total_elements * random_percentage)
    assert total_elements > num_masked_elements

    # Create a 1D tensor with the specified number of masked elements and the rest unmasked
    mask_1d = torch.cat((torch.ones(num_masked_elements), torch.zeros(total_elements - num_masked_elements)))

    # Shuffle the 1D tensor to randomize the masked element positions
    rand_mask = mask_1d[torch.randperm(total_elements)].to(device)[None].bool() # [1, l]
    rand_mask = rand_mask.expand(iterate_num, -1)

    # [n, l], true for replacing random tokens
    rand_tokens = torch.randint(low=0, high=len(tokenizer), size=[iterate_num, base_tokens.shape[1]]).to(device)
    icl_tokens = base_tokens.expand(iterate_num, base_tokens.shape[-1])
    new = icl_tokens.clone()
    new[rand_mask] = rand_tokens[rand_mask] # [n, l]
    inputs = torch.cat([prefix_tokens, new.reshape(1, -1)], dim=1) # [1, l_p + n*l]
    
    # shift left
    target_tokens = shift_left(base_tokens, pad=tokenizer.pad_token_id)
    # set replaced tokens with pad
    target_mask = shift_left(rand_mask[:1], pad=True)
    target_tokens[target_mask] = tokenizer.pad_token_id
    
    bsz, bl = base_tokens.shape
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, l*n, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, l*n, V]
    # start from prefix_len
    probs = probs[:, prefix_len:]
    logits = logits[:, prefix_len:]
    
    # reshape probs and logits
    split_probs = probs.reshape([bsz, iterate_num, bl, -1]).cpu()
    split_logits = logits.reshape([bsz, iterate_num, bl, -1]).cpu()
    
    # remove the last token because it's padding
    split_probs = split_probs[:, :, :-1]
    split_logits = split_logits[:, :, :-1]
    target_tokens = target_tokens[:, :-1]
    
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    
    results_dict = {
        "base_tokens": base_tokens.cpu(),
        "target_tokens": target_tokens.cpu(),
        "input_tokens": inputs.cpu(),
        "predicted_max_probs": predicted_max_probs,
        "split_logits": split_logits,
        "split_probs": split_probs,
    }
    
    return results_dict

# ret_dict = single_sentence_eval_icl(sentence_list[0]['base_sentence'], sentence_list[1]['base_sentence'], model, tokenizer, random_percentage=0.8)

def compute_stats_dict(document_path, model, tokenizer, random_ratio):
    stats_dict = {}

    cnt = 0
    with open(document_path, 'r') as f:
        documents = f.readlines()
        documents = [doc.strip() for doc in documents]
        for seleteced_sentence_id, rep_sen in tqdm.tqdm(enumerate(documents)):
            cnt += 1
            # Random put a prefix and put repetitive sentences as pseudo repetitive genetations
            # You can delete it and can also observe the self-reinforcement effect
            base_sentence = sentence_list[seleteced_sentence_id % len(sentence_list)]['base_sentence']
            ret_dict = single_sentence_eval_icl(base_sentence, rep_sen, model, tokenizer, iterate_num=101, random_percentage=random_ratio)
            try:
                stats_dict = gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer)
            except:
                import pdb; pdb.set_trace()

    return stats_dict


script_dir = "./self-reinforce-effect-icl"
data_dir = os.path.join(script_dir, "data/")

wiki_doc = os.path.join(data_dir, "wiki_sentences.txt.detok")
random_doc = os.path.join(data_dir, "random.llama.txt")
book_doc = os.path.join(data_dir, "book_large_p1.txt.sample_10k.detok.filter_1k")

docs = [wiki_doc, random_doc, book_doc]
keys = ['wiki', 'random', 'book']

model_dir = ''
model_sigs = [
    'llama-7b-hf',
    'llama-13b-hf',
    'llama-30b-hf',
    'llama-65b-hf-from-chiyu',
    # 'llama-65b-hf',
    # 'llama-65b-hf'
]

def main(mp):
    model_path = os.path.join(model_dir, mp)
    model, tokenizer = load_model_and_tokenizer(model_path)
    for i in range(0, 2):
        doc = docs[i]
        key = keys[i]
        random_ratios = [j/10 for j in range(10)]
        for rr in random_ratios[:]:
            f_path = os.path.join(data_dir, "{}_stats_{}_icl_rr-{}.test.pkl".format(mp, rr, key))
            # if os.path.exists(f_path):
                # print("Skip {}".format(f_path))
                # continue
            stat_dict = compute_stats_dict(doc, model, tokenizer, random_ratio=rr)
            with open(f_path, 'wb') as f:
                pickle.dump(stat_dict, f)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return

for mp in model_sigs[-1:]:
    main(mp)
    gc.collect()
