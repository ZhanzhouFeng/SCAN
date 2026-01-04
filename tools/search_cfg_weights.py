""" Search for the optimal cfg weights for the given model.
    First using 10k samples to find the optimal value, then run on 50k samples to report.
"""
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_c2i_ddp.py
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
sys.path.append("./")
from SCAN.dataset.builder import build_dataset
from SCAN.utils.distributed import init_distributed_mode, is_main_process
from SCAN.dataset.augmentation import center_crop_arr
from SCAN.util import instantiate_from_config, load_safetensors
from SCAN.eval.fid import compute_fid
from SCAN.model.utils import calculate_num_query_tokens_for_parallel_decoding_my,interleave_tokens


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sample_and_eval(tokenizer, gpt_model, cfg_scale, args, device, total_samples,revise_iter_num,token_num_each_revise):
    # Setup DDP:
    if args.parallel:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 0
        rank=0
        seed=0
        torch.manual_seed(seed)
        torch.cuda.set_device(device)

        


    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    if args.parallel:
        assert (
            total_samples % dist.get_world_size() == 0
        ), "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert (
            samples_needed_this_gpu % args.per_proc_batch_size == 0
        ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    else:
        samples_needed_this_gpu = total_samples


    folder_name = (
        f"{args.exp_name}-revise-{revise_iter_num}iters-with-{token_num_each_revise}tokens-{args.ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-"
        f"cfg-{cfg_scale:.2f}-seed-{args.global_seed}-num-{total_samples}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    if args.parallel:
        # dist.barrier()
        pass

    iterations = int(samples_needed_this_gpu // args.per_proc_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    if args.parallel:
        rank = dist.get_rank()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)

        global_batch_size = args.per_proc_batch_size * dist.get_world_size()
    else:
        rank =0 
        seed =0 
        torch.manual_seed(seed)
        global_batch_size = args.per_proc_batch_size


    token_num_per_iter= calculate_num_query_tokens_for_parallel_decoding_my(args.num_inference_steps,(args.image_size//16)**2)
    num_cond_tokens=1
    max_seq_length = (args.image_size//16)**2*2+num_cond_tokens
    attn_mask = torch.zeros(max_seq_length, max_seq_length).to(device)
    token_list = [num_cond_tokens] 
    token_list += [element for element in token_num_per_iter for _ in range(2)]
    mask_pointer=0
    for i in token_list:
        attn_mask[mask_pointer:, mask_pointer:mask_pointer+i] = 1
        mask_pointer+=i
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(torch.bool)



    cur_iter = 0
    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (args.per_proc_batch_size,), device=device)
        cfg_scales = (1.0, cfg_scale)
        # cfg_scales = (cfg_scale, cfg_scale)
    
        # indices = gpt_model.generate(
        #     cond=c_indices,
        #     token_order=None,
        #     cfg_scales=cfg_scales,
        #     num_inference_steps=args.num_inference_steps,
        #     temperature=args.temperature,
        #     top_k=args.top_k,
        #     top_p=args.top_p,
        # )

        revise_iter_nums=revise_iter_num
        revise_token_nums=[token_num_each_revise for i in range(revise_iter_nums)]


        # gen_indices ,before_revised_outputs= gpt_model.module.generate_embs_then_tokens(
        indices ,before_revised_outputs= gpt_model.generate_embs_then_tokens(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            token_num_per_iter=token_num_per_iter,
            mask =attn_mask,
            revise_token_nums=revise_token_nums,
        )

        samples = tokenizer.decode_codes_to_img(indices, args.image_size_eval)
    
        for i, sample in enumerate(samples):
            if args.parallel:
                index = i * dist.get_world_size() + rank + total
            else:
                index = i + rank+total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        cur_iter += 1
        # I use this line to look at the initial images to check the correctness
        # comment this out if you want to generate more
        if args.debug:
            import pdb; pdb.set_trace()

    sample_path = f"{sample_folder_dir}.npz"

    if args.parallel:
        # Make sure all processes have finished saving their samples before attempting to convert to .npz
        dist.barrier()
        if rank == 0:
            sample_path = create_npz_from_sample_folder(sample_folder_dir, total_samples)
            print("Done.")
        else:
            sample_path = f"{sample_folder_dir}.npz"
        dist.barrier()
        dist.destroy_process_group()
    else:
        sample_path = create_npz_from_sample_folder(sample_folder_dir, total_samples)
        print("Done.")

    fid, sfid, IS, precision, recall = compute_fid(args.ref_path, sample_path)
    return fid, sfid, IS, precision, recall


def main(args):
    # Setup PyTorch:
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    if args.parallel:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 0
        rank=0
        seed=0
        torch.manual_seed(seed)
        torch.cuda.set_device(device)

    config = OmegaConf.load(args.config)
    # create and load model
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    tokenizer.load_state_dict(state_dict)

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    model_weight = load_safetensors(args.gpt_ckpt)
    msg=gpt_model.load_state_dict(model_weight, strict=True)
    print(msg)
    gpt_model.eval()

    # Create folder to save samples:
    ckpt_string_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
    )
    args.ckpt_string_name = ckpt_string_name

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    if args.parallel:
        global_batch_size = n * dist.get_world_size()
        dist.barrier()
        dist.destroy_process_group()
    
    else:
        global_batch_size = n


    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples_search / global_batch_size) * global_batch_size)

    # CFG scales to be searched
    eval_results = {}
    cfg_scales_search = args.cfg_scales_search.split(",")
    cfg_scales_search = [float(cfg_scale) for cfg_scale in cfg_scales_search]
    cfg_scales_interval = float(args.cfg_scales_interval)
    cfg_scales_list = np.arange(cfg_scales_search[0], cfg_scales_search[1] + 1e-4, cfg_scales_interval)
    print(f"CFG scales to be searched: {cfg_scales_list}")

    result_file_name = (f"{args.results_path}/{args.exp_name}-{ckpt_string_name}-"
                        f"size-{args.image_size}-size-{args.image_size_eval}-search.json")



    if os.path.exists(result_file_name):
        # 打开 JSON 文件并加载内容
        with open(result_file_name, 'r') as file:
            data = json.load(file)

        # 获取所有的键
        exist_keys = data.keys()
    else:
        exist_keys =[]

    # run throught the CFG scales
    for cfg_scale in cfg_scales_list:
        for revise_iter_num in range(args.revise_iter_num_min,args.revise_iter_num_max,args.revise_iter_num_gap):
            for c in [4]:
                args.token_num_each_revise =c
                if f"{cfg_scale:.2f},revise {revise_iter_num} with {args.token_num_each_revise}" in exist_keys:
                    continue
                fid, sfid, IS, precision, recall = sample_and_eval(
                    tokenizer, gpt_model, cfg_scale, args, device, total_samples,revise_iter_num,args.token_num_each_revise)
                eval_results[f"{cfg_scale:.2f},revise {revise_iter_num} with {args.token_num_each_revise}"] = {
                    "fid": fid,
                    "sfid": sfid,
                    "IS": IS,
                    "precision": precision,
                    "recall": recall
                }
                # print(f"Eval results for CFG scale {cfg_scale:.2f}, revise {revise_iter_num} with {args.token_num_each_revise} tokens: {eval_results[f"{cfg_scale:.2f},revise {revise_iter_num} with {args.token_num_each_revise}"]}")

                with open(result_file_name, "w") as f:
                    json.dump(eval_results, f)
    
    # report the results
    # total_samples = int(math.ceil(args.num_fid_samples_report / global_batch_size) * global_batch_size)
    # optimal_cfg_scale = float(min(eval_results, key=lambda x: eval_results[x]["fid"]))
    # fid, sfid, IS, precision, recall = sample_and_eval(
    #     tokenizer, gpt_model, optimal_cfg_scale, args, device, total_samples)
    
    # print(f"Optimal CFG scale: {optimal_cfg_scale:.2f}")
    # print(f"Eval results for optimal CFG scale: {fid, sfid, IS, precision, recall}")
    # eval_results[f"{optimal_cfg_scale:.2f}-report"] = {
    #     "fid": fid,
    #     "sfid": sfid,
    #     "IS": IS,
    #     "precision": precision,
    #     "recall": recall
    # }

    # with open(result_file_name, "w") as f:
    #     json.dump(eval_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sample results
    parser.add_argument("--config", type=str, default="configs/SCAN/SCANmaskar_xl_0.7b_llamagen_dis.yaml")
    parser.add_argument("--exp-name", type=str, default='test_fid_revise20_4tokens')
    parser.add_argument("--gpt-ckpt", type=str, default='/new_shanghai/fengzheng.fzz/code/generation/SCAN-main/model.safetensors')
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input",)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--vq-ckpt", type=str, default='/new_shanghai/fengzheng.fzz/models/LlamaGen/vq_ds16_c2i.pt', help="ckpt path for vq model")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--cfg-scales-search", type=str, default="0.0, 2.0")
    parser.add_argument("--cfg-scales-search", type=str, default="1.0, 8.0")
    # parser.add_argument("--cfg-scales-interval", type=float, default=0.2)
    parser.add_argument("--cfg-scales-interval", type=float, default=1)
    parser.add_argument("--sample-dir", type=str, default="/new_shanghai/fengzheng.fzz/tmp")
    parser.add_argument("--num-inference-steps", type=int, default=88)
    parser.add_argument("--per-proc-batch-size", type=int, default=128)
    parser.add_argument("--num-fid-samples-search", type=int, default=10000)
    parser.add_argument("--num-fid-samples-report", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.02, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--ref-path", type=str, default="/new_shanghai/fengzheng.fzz/data/ImageNet/VIRTUAL_imagenet256_labeled.npz")
    # output results
    parser.add_argument("--results-path", type=str, default="./results")
    parser.add_argument("--parallel",  action="store_true", default=False)
    parser.add_argument('--revise_iter_num_min',type=int, default=0)
    parser.add_argument('--revise_iter_num_max',type=int, default=100)
    parser.add_argument('--revise_iter_num_gap',type=int, default=10)
    parser.add_argument('--token_num_each_revise',type=int, default=4)



    args = parser.parse_args()
    main(args)


