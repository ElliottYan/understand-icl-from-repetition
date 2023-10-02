import pickle
import os
import torch
import gc

from analyze_icl_rep.utils import *

def build_custom_parser(parser):
    parser.add_argument('--dataset', type=str, default='random')
    return parser

@torch.no_grad()
def single_sentence_eval_token_dependency(prefix_sentence, base_sentence, model, tokenizer, repeat_num=10, num_kept_tokens=2):
    device = 'cuda'
    # [prefix + iterative_num * icl_examples]
    # each icl_example differs from base sentence within a certain range
    # currently only deals with one sentences each time.
    base_tokens = maybe_encode_sentence(tokenizer, base_sentence)
    prefix_tokens = maybe_encode_sentence(tokenizer, prefix_sentence)
    prefix_len = prefix_tokens.shape[1]

    # Remove possible bos
    if base_tokens[:, 0].cpu().item() == tokenizer.bos_token_id:
        base_tokens = base_tokens[:, 1:]
    # NOTE: if the tokenizer adds eos, should deal with that.

    # Calculate the number of elements that need to be masked
    total_tokens = base_tokens.numel()
    # num_masked_elements = math.floor(total_elements * random_percentage)
    # assert total_elements > num_masked_elements
    num_kept_tokens = min(num_kept_tokens, total_tokens)

    # Create a 1D tensor with the specified number of masked elements and the rest unmasked
    if total_tokens > num_kept_tokens:
        mask_1d = torch.cat((torch.ones(total_tokens - num_kept_tokens), torch.zeros(num_kept_tokens))) # 0 for kept, 1 for random
    else:
        mask_1d = torch.zeros(num_kept_tokens) # 0 for kept, 1 for random

    # Shuffle the 1D tensor to randomize the masked element positions
    rand_mask = mask_1d[torch.randperm(total_tokens)].to(device)[None].bool() # [1, l]
    rand_mask = rand_mask.expand(repeat_num, -1)

    # [n, l], true for replacing random tokens
    rand_tokens = torch.randint(low=0, high=len(tokenizer), size=[repeat_num, base_tokens.shape[1]]).to(device)
    icl_tokens = base_tokens.expand(repeat_num, base_tokens.shape[-1])
    new = icl_tokens.clone()
    new[rand_mask] = rand_tokens[rand_mask] # [n, l]
    inputs = torch.cat([prefix_tokens, new.reshape(1, -1)], dim=1) # [1, l_p + n*l]
    
    # shift left
    target_tokens = shift_left(base_tokens, pad=tokenizer.pad_token_id)
    # set replaced tokens with pad
    target_mask = shift_left(rand_mask[:1], pad=True)
    target_tokens[target_mask] = tokenizer.pad_token_id

    # get full targets and index_mask
    t = (torch.arange(repeat_num)+1)[:, None].to(rand_mask.device) # [N, 1]
    index_mask = (~rand_mask) * t # [N, l]
    full_index_mask = torch.cat([torch.zeros_like(prefix_tokens), index_mask.reshape(1, -1)], dim=1) # [1, L]
    full_index_mask = shift_left(full_index_mask, pad=0)
    full_target_tokens = shift_left(inputs, pad=tokenizer.pad_token_id)
    
    bsz, bl = base_tokens.shape
    # duplicate several times
    outputs = model(inputs)
    logits = outputs.logits # [bsz, pl+l*n, V]
    probs = torch.softmax(logits, dim=-1) # [bsz, pl+l*n, V]
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
    
    # probs for shifted inputs
    # shifted_inputs = shift_left(inputs, pad=tokenizer.pad_token_id).unsqueeze(-1) # [bsz, pl+l*n, 1]
    # shifted_probs = torch.gather(probs, dim=2, index=shifted_inputs[:, :, None]) # [bsz, pl+l*n, 1]
    # mask out targets?
    # import pdb; pdb.set_trace()
        # "shifted_probs": shifted_probs.cpu(),
    
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
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = data_dir

    docs, keys = get_data_file(data_dir, model_sigs)
    
    def process(mp):
        model_path = os.path.join(model_dir, mp)
        model, tokenizer = load_model_and_tokenizer(model_path)
        for i in range(len(docs)):
            doc = docs[i]
            key = keys[i]
            if key != args.dataset: continue
            num_kept_tokens = [1,2,3,4]
            for kt in num_kept_tokens[:]:
                print((key, kt))
                f_path = os.path.join(output_dir, "{}_stats_{}_icl_kt-{}.test.pkl".format(mp, kt, key))
                stat_dict = compute_stats_dict(single_sentence_eval_token_dependency, doc, model, tokenizer, num_kept_tokens=kt, repeat_num=args.repeat_num)
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
    parser = build_custom_parser(parser)
    args = parser.parse_args()

    main(args)