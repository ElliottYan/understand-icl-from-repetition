from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import tqdm
import json
import os
import argparse
from utils import *

@torch.no_grad()
def single_sentence_eval(prefix_sentence, base_sentence, model, tokenizer, repeat_num=10):
    device = 'cuda'
    # [prefix + iterative_num * base_sentence]
    # currently only deals with one sentences each time.
    base_tokens = maybe_encode_sentence(tokenizer, base_sentence)
    prefix_tokens = maybe_encode_sentence(tokenizer, prefix_sentence)
    prefix_len = prefix_tokens.shape[1]

    # NOTE: if the tokenizer adds eos, should deal with that.
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
        
    # shift left
    target_tokens = shift_left(base_tokens, pad=tokenizer.pad_token_id)
    bsz, bl = base_tokens.shape
    inputs = torch.cat([prefix_tokens,] + [base_tokens, ]* repeat_num, dim=1)

    t = (torch.arange(repeat_num)+1)[:, None].to(inputs.device) # [N, 1]
    t = t.expand(-1, bl) # [N, l]
    full_index_mask = torch.cat([torch.zeros_like(prefix_tokens), t.reshape(1, -1)], dim=1) # [1, L]
    full_index_mask = shift_left(full_index_mask, pad=0)
    full_target_tokens = shift_left(inputs, pad=tokenizer.pad_token_id)
    
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, l*n, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, l*n, V]
    origin_probs = probs.clone()
    
    # start from prefix_len
    probs = probs[:, prefix_len:]
    logits = logits[:, prefix_len:]
    
    # reshape probs and logits
    split_probs = probs.reshape([bsz, repeat_num, bl, -1]).cpu()
    split_logits = logits.reshape([bsz, repeat_num, bl, -1]).cpu()
    
    # remove the last token because it's padding
    split_probs = split_probs[:, :, :-1]
    split_logits = split_logits[:, :, :-1]
    target_tokens = target_tokens[:, :-1]
    
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    
    results_dict = {
        "base_tokens": base_tokens.cpu(),
        # "target_tokens": target_tokens.cpu(),
        "target_tokens": None,
        "input_tokens": inputs.cpu(),
        "predicted_max_probs": predicted_max_probs,
        "split_logits": split_logits,
        "split_probs": split_probs,
        "full_probs": origin_probs.cpu(),
        "full_targets": full_target_tokens.cpu(),
        "full_index_mask": full_index_mask.cpu(),
    }
    
    return results_dict

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, "data/")

    model_dir, model_sigs = get_model_path(args.model_family, args.model_max_size, args.model_min_size)
    print(model_sigs)

    docs, keys = get_data_file(data_dir, model_sigs)
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = data_dir

    for mp in model_sigs[:]:
        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        for i in range(len(docs)):
            doc = docs[i]
            key = keys[i]
            stat_dict = compute_stats_dict(single_sentence_eval, doc, model, tokenizer, repeat_num=args.repeat_num)
            f_path = os.path.join(output_dir, "{}_stats_{}.pkl".format(mp, key))
            if os.path.exists(f_path):
                print("Skip {}".format(f_path))
                continue
            with open(f_path, 'wb') as f:
                pickle.dump(stat_dict, f)

if __name__ == '__main__':
    parser = build_common_parser()
    args = parser.parse_args()

    main(args)