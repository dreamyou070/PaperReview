import os, pdb
from glob import glob
import argparse
from PIL import Image
from lavis.models import load_model_and_preprocess
from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler
import sys
import numpy as np
import torch
import torch.nn.functional as F
from random import randrange
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffuser import DDIMScheduler
from diffuser.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffuser.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffuser import DiffusionPipeline
from utils.base_pipeline import BasePipeline
from utils.cross_attention import prep_unet
import torch
from diffuser.models.attention import CrossAttention
from tqdm.auto import tqdm


def auto_corr_loss(x, random_shift=True):
    B, C, H, W = x.shape
    assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = randrange(noise.shape[2] // 2)
            else:
                roll_amount = 1
            reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss

def kl_divergence(x):
    _mu = x.mean()
    _var = x.var()
    return _var + _mu ** 2 - 1 - torch.log(_var + 1e-7)

class SyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if not is_cross :
            q_src, q_x, q_y = query.chunk(3)
            k_src, k_x, k_y = key.chunk(3)
            sim_src = torch.einsum("b i d, b j d -> b i j", q_src, k_src) * self.scale  # [32, pix_num, 77]
            sim_x = torch.einsum("b i d, b j d -> b i j", q_x, k_src) * self.scale  # [32, pix_num, 77]
            sim_y = torch.einsum("b i d, b j d -> b i j", q_y, k_src) * self.scale  # [32, pix_num, 77]
            sim = torch.cat([sim_src, sim_x, sim_y])

        if is_cross :
            q_src, q_x, q_y = query.chunk(3)
            k_src, k_x, k_y = key.chunk(3)
            sim_src = torch.einsum("b i d, b j d -> b i j", q_src, k_src) * self.scale  # [32, pix_num, 77]
            sim_x = torch.einsum("b i d, b j d -> b i j", q_x, k_x) * self.scale  # [32, pix_num, 77]
            sim_y = torch.einsum("b i d, b j d -> b i j", q_y, k_y) * self.scale  # [32, pix_num, 77
            sim = torch.cat([sim_src, sim_x, sim_y])

        attn = sim.softmax(dim=-1)  # [32, pix_num, 77]
        if not is_cross:
            attn_src, attn_x, attn_y = attn.chunk(3)
            v_src, v_x, v_y = v.chunk(3)
            out_src = torch.einsum("b i j, b j d -> b i d", attn_src, v_src)
            out_x = torch.einsum("b i j, b j d -> b i d", attn_x, v_x)
            out_y = torch.einsum("b i j, b j d -> b i d", attn_y, v_y)
            # out_y = out_x
            # out_t = out_angle
            out = torch.cat([out_src, out_x, out_y])
        else:
            attn_src, attn_x, attn_y = attn.chunk(3)
            v_src, v_x, v_y = v.chunk(3)
            out_src = torch.einsum("b i j, b j d -> b i d", attn_src, v_src)
            out_x = torch.einsum("b i j, b j d -> b i d", attn_x, v_x)

            # attn_y = torch.einsum("b p d, b l d -> b p l", out_x, 1 / v_y)
            # out_y = torch.einsum("b i j, b j d -> b i d", attn_y, v_y)
            out_y = out_x
            # out_t = out_angle
            out = torch.cat([out_src, out_x, out_y])
        out = self.batch_to_head_dim(out)
        # linear proj
        hidden_states = attn.to_out[0](out)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def main(args) :

    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 1. environment')
    device = args.device
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 2. saving folder')
    inversion_folder = os.path.join(args.results_folder, "inversion")
    os.makedirs(inversion_folder, exist_ok=True)
    prompt_folder = os.path.join(args.results_folder, "prompt")
    os.makedirs(prompt_folder, exist_ok=True)
    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 3. load the BLIP model')
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption",model_type="base_coco",
                                                              is_eval=True, device=torch.device(device))
    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 4. make the DDIM inversion pipeline')
    pipe = DiffusionPipeline.from_pretrained(args.model_path,torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    print(f' (4.1) unet')
    unet = pipe.unet
    for name, params in unet.named_parameters():
        if 'attn2' in name: params.requires_grad = True
        else: params.requires_grad = False
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_processor(SyCrossAttnProcessor())

    print(f' (4.2) scheduler & classifier free & time-steps')
    guidance_scale = args.guidance_scale
    num_inversion_steps = args.num_inversion_steps
    do_classifier_free_guidance = guidance_scale > 1.0
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inversion_steps, device=device)
    timesteps = scheduler.timesteps

    print(f' (4.3) vae')
    vae = pipe.vae

    print(f' (4.4) tokenizer and text encoder')
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 5. input images')
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]
    # ------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 6. inversion')
    for img_path in l_img_paths:
        bname = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        print(f' (6.1) generating captino from BLIP model')
        _image = vis_processors["eval"](img).unsqueeze(0).to(device)
        prompt_str = model_blip.generate({"image": _image})[0]
        print(f' (6.3) inverting')
        x0 = np.array(img) / 255
        x0 = torch.from_numpy(x0).type(torch_dtype).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
        x0 = (x0 - 0.5) * 2.
        with torch.no_grad():
            x0_enc = vae.encode(x0).latent_dist.sample().to(device, torch_dtype)
        latents = x0_enc = 0.18215 * x0_enc

        def decode_latents(latents):
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            return image

        with torch.no_grad():
            x0_dec = decode_latents(x0_enc.detach())

        def numpy_to_pil(images):
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                pil_images = [Image.fromarray(image) for image in images]
            return pil_images

        image_x0_dec = numpy_to_pil(x0_dec)

        print(f' (6.4) prompt encoding')
        num_images_per_prompt = args.num_images_per_prompt
        batch_size = 1
        with torch.no_grad():
            text_input_ids = tokenizer(prompt_str,padding="max_length",max_length=tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt", ).input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), )[0]
            uncond_input = tokenizer([""],padding="longest", return_tensors="pt").input_ids
            negative_prompt_embeds = text_encoder(uncond_input.to(device),)[0]
            if do_classifier_free_guidance:
                seq_len = negative_prompt_embeds.shape[1]
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        print(f' (6.6) inversion argument')
        lambda_ac: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_ac_rolls: int = 5,

        print(f' (6.7) do the inversion')
        num_warmup_steps = len(timesteps) - num_inversion_steps * scheduler.order
        progress_bar = tqdm(total=num_inversion_steps)
        with progress_bar :
            for i, t in enumerate(timesteps.flip(0)[1:-1]):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                e_t = noise_pred
                for _outer in range(num_reg_steps):
                    if lambda_ac > 0:
                        for _inner in range(num_ac_rolls):
                            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                            l_ac = auto_corr_loss(_var)
                            l_ac.backward()
                            _grad = _var.grad.detach() / num_ac_rolls
                            e_t = e_t - lambda_ac * _grad
                    if lambda_kl > 0:
                        _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                        l_kld = kl_divergence(_var)
                        l_kld.backward()
                        _grad = _var.grad.detach()
                        e_t = e_t - lambda_kl * _grad
                    e_t = e_t.detach()
                noise_pred = e_t
                latents = scheduler.step(noise_pred, t, latents, reverse=True).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
        x_inv = latents.detach().clone()
        image = decode_latents(latents.detach())
        image = numpy_to_pil(image)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # step 1. environment
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--use_float_16', action='store_true')
    # step 2. saving folder
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    # step 3. load the BLIP model
    # step 4. make the DDIM inversion pipeline
    parser.add_argument('--model_path', type=str, default='../../pretrained_stable_diffusion/stable-diffusion-v1-4')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inversion_steps', type=int, default=50)
    # step 5. input images
    parser.add_argument('--input_image', type=str, default='assets/test_images/cats/cat_1.png')
    # step 6. inversion
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    args = parser.parse_args()
    main(args)
