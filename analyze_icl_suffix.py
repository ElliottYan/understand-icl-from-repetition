import torch
from utils import *
import pickle
import gc

@torch.no_grad()
def single_sentence_eval_phrase(prefix_sentence, base_sentence, model, tokenizer, repeat_num=10, suffix_length=1):
    device = 'cuda'
    # [prefix + iterative_num * icl_examples]
    # each icl_example differs from base sentence at the prefix
    # TODO: support index input that do not need to encode again.
    
    base_tokens = maybe_encode_sentence(tokenizer, base_sentence)
    prefix_tokens = maybe_encode_sentence(tokenizer, prefix_sentence)

    # Remove possible bos
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
    # NOTE: if the tokenizer adds eos, should deal with that.

    prefix_len = prefix_tokens.shape[1]
    
    suffix_length = min(suffix_length, base_tokens.shape[-1])
    base_suffix = base_tokens[:, -suffix_length:] # [1, sl]
    # number of tokens to generate
    random_length = max(base_tokens.shape[-1] - suffix_length, 0)
    # generate random tokens for each repeats
    base_suffix = base_suffix.expand([repeat_num, -1]) # [n, sl]
    # NOTE: now we keep the length of all repeats unchanged
    if random_length <= 0:
        new = base_suffix
        pass
    else:
        random_prefix = torch.randint(low=0, high=len(tokenizer), size=[repeat_num, random_length]).to(device) # [n, L-pl]
        # new = torch.cat([base, random_prefix], dim=1) # [n, L]
        new = torch.cat([random_prefix, base_suffix], dim=1) # [n, L]
    rand_mask = torch.ones_like(new)
    rand_mask[:, -suffix_length:] = 0 # 1 for random, 0 for kept tokens

    inputs = torch.cat([prefix_tokens, new.reshape(1, -1)], dim=1) # [1, l_p + n*l]
    
    # shift left
    target_tokens = shift_left(base_tokens, pad=tokenizer.pad_token_id)
    # set replaced tokens with pad
    target_mask = shift_left(rand_mask[:1], pad=True).bool()
    target_tokens[target_mask] = tokenizer.pad_token_id
    
    # get full targets and index_mask
    t = (torch.arange(repeat_num)+1)[:, None].to(rand_mask.device) # [N, 1]
    index_mask = (1-rand_mask) * t # [N, l]
    full_index_mask = torch.cat([torch.zeros_like(prefix_tokens), index_mask.reshape(1, -1)], dim=1) # [1, L]
    full_index_mask = shift_left(full_index_mask, pad=0)
    full_target_tokens = shift_left(inputs, pad=tokenizer.pad_token_id)
 
    bsz, bl = base_tokens.shape
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


# def compute_stats_dict(document_path, model, tokenizer, suffix_length, repeat_num=101):
#     stats_dict = {}

#     cnt = 0
#     documents = read_file(document_path)
#     for seleteced_sentence_id, rep_sen in tqdm.tqdm(enumerate(documents)):
#         cnt += 1
#         # Random put a prefix and put repetitive sentences as pseudo repetitive genetations
#         # You can delete it and can also observe the self-reinforcement effect
#         base_sentence = sentence_list[seleteced_sentence_id % len(sentence_list)]['base_sentence']
#         ret_dict = single_sentence_eval_phrase(base_sentence, rep_sen, model, tokenizer, iterate_num=repeat_num, suffix_length=suffix_length)
#         stats_dict = gather_and_update_all_metrics(stats_dict, ret_dict, tokenizer)
#         pass

#     return stats_dict


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
        
    def process(mp):
        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        for i in range(0, len(docs)):
            doc = docs[i]
            key = keys[i]
            if key != 'random':
                continue
            # random_ratios = [i/10 for i in range(10)]
            # suffix_length = [i for i in range(2, 10)]
            # suffix_length = [3,]
            suffix_length = [2,3,4,5]
            for sl in suffix_length[:]:
                f_path = os.path.join(output_dir, "{}_stats_{}_icl_sl-{}.test.pkl".format(mp, sl, key))
                # if os.path.exists(f_path):
                #     print("Skip {}".format(f_path))
                #     continue
                stat_dict = compute_stats_dict(single_sentence_eval_phrase, doc, model, tokenizer, suffix_length=sl, repeat_num=args.repeat_num)
                with open(f_path, 'wb') as f:
                    pickle.dump(stat_dict, f)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return

    for mp in model_sigs[:]:
        process(mp)
        gc.collect()
        
if __name__ == '__main__':
    parser = build_common_parser()
    args = parser.parse_args()

    main(args)