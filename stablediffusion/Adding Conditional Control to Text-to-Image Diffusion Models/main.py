from cldm.model import create_model, load_state_dict
import argparse
from annotator.midas import MidasDetector
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import cv2
import einops
import random
from pytorch_lightning import seed_everything
import torch
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import noise_like
import os
def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        #print(f'alpha_channel : {}')
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def main(args) :

    print(f'\n Step 1. model calling')
    print(f' (1.1) total model and pretrained pth') # LDM
    model = create_model(args.model_config_dir).to(args.device)
    model.load_state_dict(load_state_dict(args.resume_path, location='cuda'))
    model = model.to(args.device)
    print(f' (1.1.1) unet model')
    unet = model.model.diffusion_model
    print(f' (1.1.2) controlnet')
    control_model = model.control_model
    print(f' (1.1.3) image encoder, vae')
    vae = model.first_stage_model
    print(f' (1.1.4) text encoder')
    text_encoder = model.cond_stage_model
    print(f'\n (1.2) auxiliary model for depth map')
    auxiliary_model = MidasDetector()
    print(f'\n (1.3) sampler')
    ddim_sampler = DDIMSampler(model)

    print(f'\n\n Step 2. init image mask map')
    with torch.no_grad():
        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.1) image condition (depthmap)')
        input_image_dir = args.input_image_dir
        pil_img = Image.open(input_image_dir).convert('RGB')
        input_image = np.array(pil_img)
        depth_map, _ = auxiliary_model(resize_image(input_image, args.depthmap_resolution))
        depth_map = HWC3(depth_map) # RGB map
        img = resize_image(input_image, args.image_resolution)
        H, W, C = img.shape
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR) # nparray type
        control = torch.from_numpy(depth_map.copy()).float().to(args.device) / 255.0
        num_samples = 1
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.2) text condition')
        prompt = 'my bag '
        text_embedding = model.get_learned_conditioning([prompt + ', ' + args.a_prompt] * num_samples) # Batch, 77, 768
        null_text_embedding = model.get_learned_conditioning([args.n_prompt] * num_samples)            # Batch, 77, 768

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.3) for classifier free guidance')
        cond = {"c_concat": [control],   "c_crossattn": [text_embedding]}
        un_cond = {"c_concat": [control],"c_crossattn": [null_text_embedding]}

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.4) make initial random ( 8 smaller than initial image) ')
        seed = random.randint(0, 65535)
        seed_everything(seed)
        shape = (args.batch_size, 4, H // 8, W // 8)
        model.control_scales = [args.strength] * 13
        latent = torch.randn(shape, device = args.device)
        intermediates = {'x_inter' : latent, 'pred_x0' : latent}

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.5) sampling argument (not ddpm but ddim which means shorter inference length')
        ddim_sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0, verbose=False)
        noising_timesteps = ddim_sampler.ddim_timesteps
        denoising_timesteps = np.flip(noising_timesteps)
        total_steps = denoising_timesteps.shape[0]
        alphas = ddim_sampler.ddim_alphas
        alphas_prev = ddim_sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
        sigmas = ddim_sampler.ddim_sigmas
        print(f' sigmas : {sigmas} -> it is always 0 because it is not ddpm but ddim')

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.6) other inference argument')
        unconditional_guidance_scale = 9

        # -------------------------------------------------------------------------------------------------------------------- #
        print(f' (2.7) inference')
        iterator = tqdm(denoising_timesteps, desc='DDIM Sampler', total=total_steps)
        save_base_folder = args.save_base
        os.makedirs(save_base_folder, exist_ok=True)
        control_scales = ddim_sampler.model.control_scales

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((1,), step, device=args.device, dtype=torch.long)
            #################################################### Key Point ####################################################
            print(f' - model output')
            condition_controls = control_model(x=latent, hint=torch.cat(cond['c_concat'], 1), timesteps=ts,context=torch.cat(cond['c_crossattn'], 1))
            condition_con = []
            for control_scale, condition_control in zip(control_scales, condition_controls) :
                condition_con.append(control_scale * condition_control)
            condition_output = unet(x=latent, timesteps=ts, context = torch.cat(cond['c_crossattn'], 1), control = condition_con)
            # ------------------------------------------------------------------------------------------------------------------------------------------
            uncondition_controls = control_model(x=latent,
                                                 hint=torch.cat(un_cond['c_concat'], 1),
                                                 timesteps=ts,
                                                 context=torch.cat(un_cond['c_crossattn'], 1))
            un_condition_con = []
            for control_scale, uncondition_control in zip(control_scales, uncondition_controls):
                un_condition_con.append(control_scale * uncondition_control)
            uncondition_output = unet(x=latent, timesteps=ts, context=torch.cat(un_cond['c_crossattn'], 1),
                                      control=un_condition_con)
            model_output = uncondition_output + unconditional_guidance_scale * (condition_output - uncondition_output)
            print(f' - current and prev timestep parameter to prev_step (denoising)')
            a_t = torch.full((args.batch_size, 1, 1, 1), alphas[index], device=args.device)
            a_prev = torch.full((args.batch_size, 1, 1, 1), alphas_prev[index], device=args.device)
            sigma_t = torch.full((args.batch_size, 1, 1, 1), sigmas[index], device=args.device)
            sqrt_one_minus_at = torch.full((args.batch_size, 1, 1, 1), sqrt_one_minus_alphas[index],device=args.device)
            if ddim_sampler.model.parameterization == "v":
                e_t = ddim_sampler.model.predict_eps_from_z_and_v(latent, ts, model_output)
            else:
                e_t = model_output
            if ddim_sampler.model.parameterization != "v":
                pred_x0 = (latent - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = ddim_sampler.model.predict_start_from_z_and_v(latent, ts, model_output)
            direction_pointing_to_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            random_noise = sigma_t * noise_like(latent.shape, args.device, args.repeat_noise) * args.temperature
            print(f' - paper (12) equation, current latent')
            latent = a_prev.sqrt() * pred_x0 + direction_pointing_to_xt + random_noise
            ################################################## timestep-wise image sabe ###############################################
            x_samples = model.decode_first_stage(latent)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            x_samples = np.squeeze(x_samples)
            img = Image.fromarray(x_samples.astype(np.uint8))
            latent_dir = os.path.join(save_base_folder, f'latent_{ts.item()}.jpg')
            img.save(latent_dir)
            print(f' -------------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    # 1
    parser.add_argument('--model_config_dir', default=r'./models/cldm_v15.yaml', type=str)
    parser.add_argument('--resume_path', default=r'./ControlNet/pretrained_models/control_sd15_depth.pth', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    # 2
    parser.add_argument('--input_image_dir', default=r'test_imgs/bag.png', type=str)
    parser.add_argument('--depthmap_resolution', default=384, type=int)
    parser.add_argument('--image_resolution', default=512, type=int)
    # 2.2
    parser.add_argument('--a_prompt', default='best quality', type=str)
    parser.add_argument('--n_prompt', default='ugly', type=str)
    parser.add_argument('--strength', default=1, type=int)
    parser.add_argument('--save_base', default='save_base', type=str)
    # 2.4
    parser.add_argument('--batch_size', default=1, type=int)
    # 2.5
    parser.add_argument('--ddim_steps', default=20, type=int)
    # 2.7
    parser.add_argument('--repeat_noise', action='store_true')
    parser.add_argument('--temperature', default = 1, type = int)
    args = parser.parse_args()
    main(args)
