import wandb
import argparse
import gc
import math
import os
import random
import time
from networks import lora
import json
from networks import lora_block_training
from multiprocessing import Value
from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
import library.train_util as train_util
from library.train_util import (DreamBoothDataset,DatasetGroup)
import library.config_util as config_util
from library.config_util import (ConfigSanitizer,BlueprintGenerator,)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    pyramid_noise_like,
    apply_noise_offset,
    scale_v_prediction_loss_like_noise_prediction,)
import time
from datetime import date
import datetime
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import re
from setproctitle import *
from library.train_util import _load_target_model
from library import model_util

setproctitle('parksooyeon')

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

def sample_images_blockwise_single(args: argparse.Namespace,
                                   epoch, steps, device, vae,
                                   tokenizer, text_encoder, unet,
                                   block_wise_save_folder,
                                   prompt_replacement=None):

    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return
    if args.sample_every_n_epochs is not None:
        if epoch is None or epoch % args.sample_every_n_epochs != 0:
            return
    else:
        if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
            return
    print(f"\ngenerating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        print(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return
    org_vae_device = vae.device  # CPUにいるはず
    vae.to(device)
    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
        #prompts = [line.replace('<TOK>', args.target_character) for line in lines]
        #sub_prompts = [line.replace('<TOK>', args.compare_character) for line in lines]
        #prompts = prompts + sub_prompts
        prompts = lines
    sched_init_args = {}
    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"
    scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                            beta_start=SCHEDULER_LINEAR_START,
                            beta_end=SCHEDULER_LINEAR_END,
                            beta_schedule=SCHEDLER_SCHEDULE,
                            **sched_init_args,)

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    pipeline = StableDiffusionLongPromptWeightingPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_skip=args.clip_skip,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,)
    pipeline.to(device)
    sample_save_dir = os.path.join(block_wise_save_folder, "sample")
    os.makedirs(sample_save_dir, exist_ok=True)
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prompt_args = prompt.split(" --")
            prompt = prompt_args[0]
            negative_prompt = 'low quality, worst quality, ba anatomy, bad composition, poor, ow effort, bad hands, missing fingers, cropped',
            sample_steps = 30
            width = 512
            height = 768
            scale = 7
            seed = args.seed
            for parg in prompt_args:
                try:
                    m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                    if m:
                        width = int(m.group(1))
                        continue
                    m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                    if m:
                        height = int(m.group(1))
                        continue
                    m = re.match(r"d (\d+)", parg, re.IGNORECASE)
                    if m:
                        seed = int(m.group(1))
                        continue
                    m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                    if m:  # steps
                        sample_steps = max(1, min(1000, int(m.group(1))))
                        continue
                    m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                    if m:  # scale
                        scale = float(m.group(1))
                        continue

                    m = re.match(r"n (.+)", parg, re.IGNORECASE)
                    if m:  # negative prompt
                        negative_prompt = m.group(1)
                        continue
                except ValueError as ex:
                    print(f"Exception in parsing / 解析エラー: {parg}")
                    print(ex)
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            if prompt_replacement is not None:
                prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
                if negative_prompt is not None:
                    negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])
            height = max(64, height - height % 8)  # round to divisible by 8
            width = max(64, width - width % 8)  # round to divisible by 8
            image = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=sample_steps,
                guidance_scale=scale,
                negative_prompt=negative_prompt,
            ).images[0]
            import numpy as np

            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            seed_suffix = "" if seed is None else f"_{seed}"
            p_split = prompt.split(' ')
            p_split = '_'.join(p_split)
            img_filename = (f"{'' if args.output_name is None else args.output_name + '_'}{ts_str}_{num_suffix}_{p_split}.png")
            epoch_dir = os.path.join(sample_save_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            image.save(os.path.join(epoch_dir, img_filename))
    del pipeline
    torch.cuda.empty_cache()
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)


def generate_step_logs(args: argparse.Namespace,
                       current_loss, avr_loss, lr_scheduler,
                       keys_scaled=None, mean_norm=None, maximum_norm=None
):
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/average_key_norm"] = mean_norm
        logs["max_norm/max_key_norm"] = maximum_norm

    lrs = lr_scheduler.get_last_lr()

    if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
        if args.network_train_unet_only:
            logs["lr/unet"] = float(lrs[0])
        elif args.network_train_text_encoder_only:
            logs["lr/textencoder"] = float(lrs[0])
        else:
            logs["lr/textencoder"] = float(lrs[0])
            logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

        if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():  # tracking d*lr value of unet.
            logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
    else:
        idx = 0
        if not args.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                logs[f"lr/d*lr/group{i}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"])
    return logs

def train(args):

    print(f'step 1. start of experiment')
    session_id = random.randint(0, 2 ** 32)
    training_started_at = time.time()
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None
    use_user_config = args.dataset_config is not None

    print(f' step 2. set seed')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f' (1.2) save base folder')
    today = date.today()
    unique = str(time.time()).replace('.', '_')
    mother, cur_dir = os.path.split(args.output_dir)
    save_base_folder = os.path.join(mother, f'{cur_dir}_{today}_{unique}')
    os.makedirs(save_base_folder, exist_ok=True)
    with open(os.path.join(save_base_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # -------------------------------------------------------------------------------------------------------- #
    print(f'step 2. lora weight setting')
    net_kwargs = {}
    text_encoder_num = 1
    unet_num = 16
    full_weight = 0
    for i in range(text_encoder_num + unet_num):
        weight = 2 ** i
        full_weight += weight
    lora_block_weights = []
    for wei in range(full_weight+1):
        if wei > 0 :
            wei = bin(wei)[2:]
            wei = wei.zfill(17)
            wei_list = [int(w) for w in wei]
            w_1_num = 0
            for w in wei_list :
                if w == 1 :
                    w_1_num += 1
            if w_1_num == args.trg_num_1 :
                lora_block_weights.append(wei_list)

    # -------------------------------------------------------------------------------------------------------- #
    print(f'step 3. partial training')
    for index, lora_block_training_options in enumerate(lora_block_weights) :
        lora_block_name = ''
        for i in lora_block_training_options :
            lora_block_name += str(i)
        block_wise_save_folder = os.path.join(save_base_folder, f'block_wise_{lora_block_name}')
        os.makedirs(block_wise_save_folder, exist_ok=True)
        print(f' ({index}.1) tokenizer and dataset preparing')
        tokenizer = train_util.load_tokenizer(args)
        #blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, True))
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        """
        user_config = {'datasets': [{'subsets': [{'num_repeats': args.num_repeats,
                                                  'image_dir': args.image_dir}]}],
                       'general': {'resolution': args.resolution,
                                   'shuffle_caption': args.shuffle_caption,
                                   'keep_tokens': args.keep_tokens,
                                   'flip_aug': args.flip_aug,
                                   'caption_extension': args.caption_extension,
                                   'enable_bucket': args.enable_bucket,
                                   'bucket_reso_steps': args.bucket_reso_steps,
                                   'bucket_no_upscale': args.bucket_no_upscale,
                                   'min_bucket_reso': args.min_bucket_reso,
                                   'max_bucket_reso': args.max_bucket_reso}}
        """
        user_config = {
            "datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir,
                                                                                               args.reg_data_dir)}]}
        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        print(f' ({index}.2) dataloader')
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)
        if cache_latents:
            assert (train_dataset_group.is_latent_cacheable()),\
                    "when caching latents, either color_aug or random_crop cannot be used"



        lora_block_training_options = [bool(i) for i in lora_block_training_options]

        print(f' ({index}.3) device and dtype')
        device = args.device
        weight_dtype, save_dtype = train_util.prepare_dtype(args)

        print(f' ({index}.4) diffusion model')
        text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(args,weight_dtype,device,)
        gc.collect()
        torch.cuda.empty_cache()

        text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(args, weight_dtype, device)
        model_version = model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)
        if cache_latents:
            vae.to(device, dtype=weight_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae,
                                                  args.vae_batch_size,
                                                  args.cache_latents_to_disk,
                                                  1)
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print(f' ({index}.5) netework')
        network = lora_block_training.create_network(lora_block_training_options,
                                                     args.network_dim,
                                                     args.network_alpha,
                                                     vae, text_encoder, unet,
                                                     neuron_dropout=args.network_dropout,
                                                     **net_kwargs)
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            text_encoder.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()  # may have no effect

        print(f' ({index}.6) preparing optimizer, data loader etc')
        trainable_params = network.prepare_optimizer_params(args.text_encoder_lr,
                                                            args.unet_lr,
                                                            args.learning_rate,
                                                            blockwise_lr=False)
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        print(f' ({index}.7) preparing data loader etc')
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1
        train_dataloader = torch.utils.data.DataLoader(train_dataset_group,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       collate_fn=collater,
                                                       num_workers=n_workers,
                                                       persistent_workers=args.persistent_data_loader_workers,)
        args.max_train_steps = args.max_train_epochs * math.ceil(len(train_dataloader)
                                                                 / args.gradient_accumulation_steps)
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        print(f' ({index}.8) lr scheduler')
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, 1)

        print(f' ({index}.9) model to device and precision')
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16' "
            print("enabling full fp16 training.")
            network.to(weight_dtype)
        unet.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        network.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        vae.eval()
        if args.gradient_checkpointing:
            unet.train()
            text_encoder.train()
            text_encoder.text_model.embeddings.requires_grad_(True)
        else:
            unet.eval()
            text_encoder.eval()
        # --------------------------------------------------------------------------------------------------
        # (1) original parameter
        parameter_gradient_magnitude_dict = {}
        for name, param in network.named_parameters():
            parameter_gradient_magnitude_dict[name] = 0

        print(f' ({index}.10) wandb project')
        wandb.init(project=f'{lora_block_name}-{today}-gradnorm-{unique}',
                   config = parameter_gradient_magnitude_dict)

        network.prepare_grad_etc(text_encoder, unet)
        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()

        print(f' ({index}.11) transform DDP after prepare (train_network here only)')
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

        print(f' ({index}.12) training')
        print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        print(f"  num epochs / epoch数: {num_train_epochs}")
        print(f"  batch size per device : {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_dim": args.network_dim,
            # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
        }
        datasets_metadata = []
        tag_frequency = {}  # merge tag frequency for metadata editor
        dataset_dirs_info = {}  # merge subset dirs for metadata editor

        for dataset in train_dataset_group.datasets:
            is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
            dataset_metadata = {
                "is_dreambooth": is_dreambooth_dataset,
                "batch_size_per_device": dataset.batch_size,
                "num_train_images": dataset.num_train_images,  # includes repeating
                "num_reg_images": dataset.num_reg_images,
                "resolution": (dataset.width, dataset.height),
                "enable_bucket": bool(dataset.enable_bucket),
                "min_bucket_reso": dataset.min_bucket_reso,
                "max_bucket_reso": dataset.max_bucket_reso,
                "tag_frequency": dataset.tag_frequency,
                "bucket_info": dataset.bucket_info,}
            subsets_metadata = []
            for subset in dataset.subsets:
                subset_metadata = {
                    "img_count": subset.img_count,
                    "num_repeats": subset.num_repeats,
                    "color_aug": bool(subset.color_aug),
                    "flip_aug": bool(subset.flip_aug),
                    "random_crop": bool(subset.random_crop),
                    "shuffle_caption": bool(subset.shuffle_caption),
                    "keep_tokens": subset.keep_tokens,}

                image_dir_or_metadata_file = None
                if subset.image_dir:
                    image_dir = os.path.basename(subset.image_dir)
                    subset_metadata["image_dir"] = image_dir
                    image_dir_or_metadata_file = image_dir

                if is_dreambooth_dataset:
                    subset_metadata["class_tokens"] = subset.class_tokens
                    subset_metadata["is_reg"] = subset.is_reg
                    if subset.is_reg:
                        image_dir_or_metadata_file = None  # not merging reg dataset
                else:
                    metadata_file = os.path.basename(subset.metadata_file)
                    subset_metadata["metadata_file"] = metadata_file
                    image_dir_or_metadata_file = metadata_file  # may overwrite
                subsets_metadata.append(subset_metadata)

                if image_dir_or_metadata_file is not None:
                    # datasets may have a certain dir multiple times
                    v = image_dir_or_metadata_file
                    i = 2
                    while v in dataset_dirs_info:
                        v = image_dir_or_metadata_file + f" ({i})"
                        i += 1
                    image_dir_or_metadata_file = v
                    dataset_dirs_info[image_dir_or_metadata_file] = {"n_repeats": subset.num_repeats,
                                                                     "img_count": subset.img_count}
            dataset_metadata["subsets"] = subsets_metadata
            datasets_metadata.append(dataset_metadata)

            for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                if ds_dir_name in tag_frequency:
                    continue
                tag_frequency[ds_dir_name] = ds_freq_for_dir

        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_keys = ["ss_network_module",
                        "ss_network_dim", "ss_network_alpha", "ss_network_args"]
        minimum_metadata = {}
        for key in minimum_keys:
            if key in metadata:
                minimum_metadata[key] = metadata[key]
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, desc="steps")
        global_step = 0
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, device)
        loss_list = []
        loss_total = 0.0
        del train_dataset_group

        # callback for step start
        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None
            # function for saving/removing
            def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
                model_save_dir = os.path.join(block_wise_save_folder,'model')
                os.makedirs(model_save_dir, exist_ok=True)
                ckpt_file = os.path.join(model_save_dir, ckpt_name)
                print(f"\nsaving checkpoint: {ckpt_file}")
                metadata["ss_training_finished_at"] = str(time.time())
                metadata["ss_steps"] = str(steps)
                metadata["ss_epoch"] = str(epoch_no)
                unwrapped_nw.save_weights(ckpt_file, save_dtype,
                                          minimum_metadata if args.no_metadata else metadata)
                if args.huggingface_repo_id is not None:
                    huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)
            def remove_model(old_ckpt_name):
                model_save_dir = os.path.join(block_wise_save_folder, 'model')
                old_ckpt_file = os.path.join(model_save_dir, old_ckpt_name)
                if os.path.exists(old_ckpt_file):
                    print(f"removing old checkpoint: {old_ckpt_file}")
                    os.remove(old_ckpt_file)

            # training loop
            for epoch in range(num_train_epochs):
                print(f"\nepoch {epoch + 1}/{num_train_epochs}")
                current_epoch.value = epoch + 1
                metadata["ss_epoch"] = str(epoch + 1)
                network.on_epoch_start(text_encoder, unet)
                for step, batch in enumerate(train_dataloader):
                    current_step.value = global_step
                    on_step_start(text_encoder, unet)
                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(device, dtype=weight_dtype)
                        else:
                            # latentに変換
                            latents = vae.encode(batch["images"].to(device, dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * 0.18215
                    b_size = latents.shape[0]

                    with torch.set_grad_enabled(train_text_encoder):
                        if args.weighted_captions:
                            encoder_hidden_states = get_weighted_text_embeddings(tokenizer,
                                                                                 text_encoder,
                                                                                 batch["captions"],
                                                                                 device,
                                                                                 args.max_token_length // 75 if args.max_token_length else 1,
                                                                                 clip_skip=args.clip_skip,)

                        else:
                            input_ids = batch["input_ids"].to(device)
                            encoder_hidden_states = train_util.get_hidden_states(args,
                                                                                 input_ids,
                                                                                 tokenizer,
                                                                                 text_encoder,
                                                                                 weight_dtype)

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents, device=latents.device)
                        if args.noise_offset:
                            noise = apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
                        elif args.multires_noise_iterations:
                            noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations,
                                                       args.multires_noise_discount)

                        # Sample a random timestep for each image
                        timesteps = torch.randint(0,
                                                  noise_scheduler.config.num_train_timesteps,
                                                  (b_size,),
                                                  device=latents.device)
                        timesteps = timesteps.long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        if args.v_parameterization:
                            # v-parameterization training
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            target = noise
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])
                        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                        loss = loss * loss_weights.to(device, dtype=weight_dtype)
                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                        # ---------------------------------------------------------------------------------------------- #
                        optimizer.zero_grad()
                        loss.retain_grad()
                        loss.backward()
                        optimizer_params_1 = optimizer.param_groups[0]['params']
                        lr_1 = optimizer.param_groups[0]['lr']
                        optimizer_params_2 = optimizer.param_groups[1]['params']
                        lr_2 = optimizer.param_groups[1]['lr']
                        optimizer_params = optimizer_params_1 + optimizer_params_2
                        gradient_dict = {}
                        param_dict = {}
                        for (name, param), opt_params in zip(network.named_parameters(), optimizer_params):
                            gradient = opt_params.grad.data
                            gradient_dict[name] = gradient
                            param_dict[name] = opt_params.data
                        wandb.log(param_dict, step=global_step)

                        optimizer.step()
                        lr_scheduler.step()
                    if args.scale_weight_norms:
                        keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                            args.scale_weight_norms, device)
                        max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None
                    progress_bar.update(1)
                    global_step += 1
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    current_loss = loss.detach().item()
                    if epoch == 0:
                        loss_list.append(current_loss)
                    else:
                        loss_total -= loss_list[step]
                        loss_list[step] = current_loss
                    loss_total += current_loss
                    avr_loss = loss_total / len(loss_list)
                    logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    if args.scale_weight_norms:
                        progress_bar.set_postfix(**{**max_mean_logs, **logs})
                    if args.logging_dir is not None:
                        logs = generate_step_logs(args, current_loss,
                                                  avr_loss, lr_scheduler, keys_scaled, mean_norm,
                                                  maximum_norm)
                        wandb.log(logs, step=global_step)

                    if global_step >= args.max_train_steps:
                        break

                if args.logging_dir is not None:
                    logs = {"loss/epoch": loss_total / len(loss_list)}
                    wandb.log(logs, step=epoch + 1)
                if args.save_every_n_epochs is not None:
                    saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, network, global_step, epoch + 1)
                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as,
                                                                          remove_epoch_no)
                        remove_model(remove_ckpt_name)

                sample_images_blockwise_single(args, epoch + 1,
                                                          global_step, device, vae, tokenizer,
                                                          text_encoder, unet, block_wise_save_folder)
            metadata["ss_training_finished_at"] = str(time.time())
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            print("model saved.")
        del network


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # step 1. session_id
    parser.add_argument("--cache_latents", action="store_false", help="cache latents to main memory to reduce VRAM usage", )
    parser.add_argument("--in_json", type=str, default=None, help="json metadata for dataset")
    parser.add_argument("--dataset_config", default=None,
                        help="config file for detail settings")
    parser.add_argument("--seed", type=int, default=None, help="random seed for training")
    parser.add_argument("--output_dir", type=str, default=r'result/lora_block_test',
                        help="directory to output trained model")

    # step 2. lora weight setting
    parser.add_argument("--trg_num_1", type=int, default=17, )

    # step 3. partial training
    parser.add_argument("--num_repeats", default=10, type=int)
    parser.add_argument("--image_dir", type=str, default=r'data/150_zaradress')
    parser.add_argument("--resolution", type=str, default='512,768')
    parser.add_argument("--shuffle_caption", action="store_false",
                        help="shuffle comma-separated caption")
    parser.add_argument("--keep_tokens", type=int, default=1,
                        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / caption", )
    parser.add_argument("--flip_aug", action="store_true",
                        help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    parser.add_argument("--caption_extension", type=str, default=".txt",
                        help="extension of caption files / caption")
    parser.add_argument("--caption_extention", type=str, default=".txt",
                        help="extension of caption files / caption")
    parser.add_argument("--enable_bucket", action="store_false",
                        help="enable buckets for multi aspect ratio training")
    parser.add_argument("--bucket_reso_steps",
                        type=int, default=64, help="steps of resolution for buckets, divisible by 8 is recommended ", )
    parser.add_argument("--bucket_no_upscale", action="store_true",
                        help="make bucket for each image without upscaling")
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument("--color_aug", action="store_true",
                        help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    # 3.2 dataloader
    # 3.3 device and dtype
    parser.add_argument("--device", type=str, default=r'cuda:0', )
    # 3.4 diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=r'pretrained/sd-v1-5-pruned-noema-fp16.safetensors')
    parser.add_argument("--mem_eff_attn", action="store_false", )
    parser.add_argument("--xformers", action="store_false",
                        help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument("--vae_batch_size", type=int, default=1,
                        help="batch size for caching latents / latentのcache時のバッチサイズ")
    parser.add_argument("--v2", action="store_true",
                        help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0")
    parser.add_argument("--v_parameterization", action="store_true",
                        help="enable v-parameterization training / v-parameterization")
    parser.add_argument("--vae", type=str, default=None, help="path to checkpoint of vae to replace ")
    parser.add_argument("--cache_latents_to_disk", action="store_true",
                        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM", )
    # 3.5
    parser.add_argument("--network_dim", type=int, default=32,
                        help="network dimensions (depends on each network)")
    parser.add_argument("--network_alpha", type=float, default=16,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="enable gradient checkpointing / grandient checkpointingを有効にする")
    # 3.6 preparing optimizer, data loader etc
    parser.add_argument("--unet_lr", type=float, default=0.0005, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=0.001,
                        help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate / 学習率")
    # 3.7 preparing data loader etc
    parser.add_argument("--max_data_loader_n_workers", type=int, default=8,
                        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)")
    parser.add_argument("--persistent_data_loader_workers", action="store_false",
                        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader ", )
    parser.add_argument("--max_train_epochs",
                        type=int, default=20, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_train_steps", type=int,
                        default=1600, help="training steps / 学習ステップ数")
    # 3.8 lr scheduler
    # 3.9 model to device and precision
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], )
    # 3.10 wandb project
    # 3.11 transform DDP after prepare (train_network here only)
    parser.add_argument("--save_n_epoch_ratio", type=int, default=None,
                        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total)", )
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="save checkpoint every N epochs")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps")
    # 3.12

    parser.add_argument("--max_token_length", type=int, default=225)
    parser.add_argument("--debug_dataset", action="store_true",
                        help="show images for debugging (do not train) ")
    parser.add_argument("--log_with", type=str, default="tensorboard", choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--logging_dir",type=str, default='log',
                        help="enable logging and output TensorBoard log to this directory",)
    parser.add_argument("--log_prefix", type=str, default='test',
                        help="add prefix for each log directory")
    parser.add_argument("--log_tracker_name", type=str,default=None, help="name of tracker to use for logging, default is scripts-specific default name",)
    parser.add_argument("--wandb_api_key", type=str, default=None, help="specify WandB API key to log in before starting training (optional).",)
    parser.add_argument("--config_file", type=str,default=None,
                        help="using .toml instead of args to pass hyperparameter ", )
    parser.add_argument("--optimizer_type", type=str,default="AdamW8bit",)
    parser.add_argument("--use_8bit_adam",action="store_true", help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adam",)
    parser.add_argument("--use_lion_optimizer", action="store_true",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,)
    parser.add_argument("--optimizer_args",type=str,default=None,nargs="*")
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument("--lr_scheduler_args", type=str,default=None,  nargs="*",)
    parser.add_argument("--lr_scheduler", type=str,default="cosine_with_restarts",)
    parser.add_argument("--lr_warmup_steps",type=int,default=40)
    parser.add_argument("--lr_scheduler_num_cycles",type=int,default=3)
    parser.add_argument("--lr_scheduler_power",type=float,default=1,)
    parser.add_argument("--save_last_n_epochs", type=int, default=4,
                        help="save last N checkpoints when saving every N epochs (remove older checkpoints)", )
    parser.add_argument("--save_last_n_epochs_state", type=int, default=None,
                        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)", )
    parser.add_argument("--save_last_n_steps", type=int, default=None,
                        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) ", )
    parser.add_argument("--save_last_n_steps_state", type=int, default=None,
                        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) ", )
    parser.add_argument("--save_state", action="store_true", help="save training state additionally ", )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")
    parser.add_argument("--clip_skip",type=int,default=1,
                        help="use output of nth layer from back of text encoder (n>=1) / text encoder",)
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata")
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, )
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--face_crop_aug_range", type=str, default=None,
                        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) ", )
    parser.add_argument("--random_crop",action="store_true",
                        help="enable random crop (for style training in face-centered crop augmentation) ",)
    parser.add_argument("--token_warmup_min", type=int,default=1,
                        help="start learning at N tags (token means comma separated strinfloatgs) ",)
    parser.add_argument("--token_warmup_step",
                        type=float,default=0,help="tag length reaches maximum on N steps ",)
    parser.add_argument("--dataset_class", type=str,default=None,
                        help="dataset class for arbitrary dataset (package.module.Class) ",)
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0) / captionをdropout",)
    parser.add_argument("--caption_dropout_every_n_epochs",type=int,default=0, help="Dropout all captions every N epochs",)
    parser.add_argument("--caption_tag_dropout_rate",type=float, default=0.0,
                        help="Rate out dropout comma separated tokens(0.0~1.0)",)
    parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images")
    parser.add_argument("--dataset_repeats", type=int, default=1,
                        help="repeat dataset when training with captions")
    parser.add_argument("--output_name", type=str, default='test_jason',
                        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    parser.add_argument("--huggingface_repo_id", type=str, default=None,
                        help="huggingface repo name to upload / huggingface")
    parser.add_argument("--huggingface_repo_type", type=str, default=None,
        help="huggingface repo type to upload / huggingface")
    parser.add_argument("--huggingface_path_in_repo",type=str,default=None,
                        help="huggingface model path to upload files / huggingface",)
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument("--huggingface_repo_visibility", type=str,default=None,
                        help="huggingface repository visibility ('public' for public, 'private' or None for private)",)
    parser.add_argument("--save_state_to_huggingface", action="store_true", help="save state to huggingface")
    parser.add_argument("--resume_from_huggingface",  action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",)
    parser.add_argument("--async_upload",  action="store_true", help="upload to huggingface asynchronously / huggingface",)
    parser.add_argument("--save_precision", type=str, default="fp16",choices=[None, "float", "fp16", "bf16"],)
    # step 5. stable diffusion
    parser.add_argument("--noise_offset",type=float,default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offset",)
    parser.add_argument("--multires_noise_iterations",type=int,default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noise",)
    parser.add_argument("--multires_noise_discount",type=float,default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount",)
    parser.add_argument("--adaptive_noise_scale", type=float,default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default)",)
    parser.add_argument("--lowram",action="store_false", help="enable low RAM optimization.",)
    # step 12. inference
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                         help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する"    )
    parser.add_argument("--sample_every_n_epochs",type=int,default=1,
                         help="generate sample images every N epochs (overwrites n_steps)",)
    parser.add_argument("--sample_prompts", type=str, default='test/test_prompt.txt',)
    parser.add_argument("--sample_sampler", type=str,default="euler_a",)
    parser.add_argument("--output_config", action="store_true", help="output command line args to given .toml file")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images" )
    parser.add_argument("--min_snr_gamma", type=float, default=0.5,)
    parser.add_argument("--scale_v_pred_loss_like_noise_pred", action="store_true",)
    parser.add_argument( "--weighted_captions",action="store_true", default=False,)
    parser.add_argument("--no_metadata", action="store_true",
                        help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument("--save_model_as",type=str,default="safetensors",choices=[None, "ckpt", "pt", "safetensors"],)
    parser.add_argument("--dim_from_weights",action="store_true")
    parser.add_argument("--scale_weight_norms",type=float,default=None,)
    parser.add_argument("--base_weights",type=str,default=None)
    parser.add_argument("--base_weights_multiplier",type=float, default=None)
    parser.add_argument("--target_character",type=str, default='striped cloth',)
    parser.add_argument("--compare_character",type=str, default='cloth',)
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)
