import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torchvision.transforms import Resize
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import einops

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)



def main(args):
    # --------------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 1. environment')
    seed_everything(args.seed)
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    # --------------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 2. model')
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")
    device = args.device
    model = model.to(device)
    print(f' (2.1) sampler')
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    print(f' (2.2) vae')
    vae = model.first_stage_model
    print(f' (2.3) text encoder')
    ref_img_encoder = model.cond_stage_model
    print(f' (2.4) unet')
    unet = model.model.diffusion_model
    # --------------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 3. save directory')
    print(f' (3.1) output directory')
    outpath = args.outdir
    os.makedirs(outpath, exist_ok=True)
    print(f' (3.2) sample_path')
    sample_path = os.path.join(outpath, "source")
    os.makedirs(sample_path, exist_ok=True)
    print(f' (3.3) result_path')
    result_path = os.path.join(outpath, "results")
    os.makedirs(result_path, exist_ok=True)
    print(f' (3.4) grid_path')
    grid_path = os.path.join(outpath, "grid")
    os.makedirs(grid_path, exist_ok=True)
    # --------------------------------------------------------------------------------------------------------------------------------
    print(f'\n step 4. inference base config')
    batch_size = args.n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size
    # --------------------------------------------------------------------------------------------------------------------------------

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # ------------------------------------------------------------------------------------------------------------------- #
                # 1.1 img
                img_p = Image.open(args.image_path).convert("RGB")
                image_tensor = get_tensor()(img_p)
                image_tensor = image_tensor.unsqueeze(0)
                # 1.2 mask
                mask = Image.open(args.mask_path).convert("L")
                mask = np.array(mask)[None, None]
                mask = 1 - mask.astype(np.float32) / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                mask_tensor = torch.from_numpy(mask)
                # 1. kwrgs (background condition, will be go with query)
                test_model_kwargs = {}
                inpaint_image = image_tensor * mask_tensor
                z_inpaint = model.encode_first_stage(inpaint_image.to(device))
                z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()

                test_model_kwargs['inpaint_image'] = z_inpaint
                b, c, h, w = z_inpaint.shape
                test_model_kwargs['inpaint_mask'] = Resize([h,w])(mask_tensor.to(device))

                from torchvision.transforms.functional import to_pil_image
                inpaint_img = to_pil_image(z_inpaint.squeeze(0)).convert('RGB')
                inpaint_img.save(f'masked_img.jpg')

                inpaint_mask_shape = test_model_kwargs['inpaint_mask'].shape

                # ------------------------------------------------------------------------------------------------------------------- #
                # 2. ref condition
                ref_p = Image.open(args.reference_path).convert("RGB").resize((224, 224))
                ref_tensor = get_tensor_clip()(ref_p)
                ref_tensor = ref_tensor.unsqueeze(0)
                ref_tensor = ref_tensor.to(device)
                uc = None
                if args.guidance_scale != 1.0:
                    uc = model.learnable_vector
                c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                c = model.proj_out(c)
                print(f'condition: {c.shape} | uncondition : {uc.shape}')

                # ------------------------------------------------------------------------------------------------------------------- #
                # 3. init latent
                shape = [args.C, args.H // args.f, args.W // args.f]
                start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

                # ------------------------------------------------------------------------------------------------------------------- #
                # 4. timestep
                sampler.make_schedule(ddim_num_steps = args.ddim_steps, ddim_eta = args.ddim_eta, verbose = False)
                alphas = sampler.ddim_alphas
                alphas_prev = sampler.ddim_alphas_prev
                sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
                sigmas = sampler.ddim_sigmas
                intermediates = {'x inter': [start_code], 'pred x0': [start_code]}
                timesteps = sampler.ddim_timesteps  # 1 -> 951
                time_range = np.flip(timesteps)     # 951 -> 1
                total_steps = timesteps.shape[0]    # 50
                print(f'timesteps : {timesteps} | time_range : {time_range} | total_steps : {total_steps}')
                iterator = tqdm(time_range, desc='Sampler', total=total_steps)
                x = start_code
                for i, step in enumerate(time_range) :

                    # 1. timestep and kwargs
                    index = total_steps - i - 1 # 49
                    ts = torch.full((args.n_samples,), step, device=device, dtype=torch.long)
                    next_step = time_range[min(i+1,total_steps-1)] # second
                    ts_next = torch.full((args.n_samples,), next_step, device=device, dtype=torch.long)

                    # 2. unet input
                    x_con = torch.cat([x, test_model_kwargs['inpaint_image'], test_model_kwargs['inpaint_mask']], dim=1) # total 9 = 4 + 4 + 1
                    x_in = torch.cat([x_con] * 2) if args.guidance_scale > 1 else x_con
                    t_in = torch.cat([ts] * 2) if args.guidance_scale > 1 else ts
                    c_in = torch.cat([uc, c])
                    e_t_uncond, e_t = unet(x_in, t_in, c_in).chunk(2)
                    e_t = e_t_uncond + args.guidance_scale * (e_t - e_t_uncond)

                    # select parameters corresponding to the currently considered timestep
                    a_t = torch.full((args.n_samples, 1, 1, 1), alphas[index], device=device)
                    a_prev = torch.full((args.n_samples, 1, 1, 1), alphas_prev[index], device=device)
                    sigma_t = torch.full((args.n_samples, 1, 1, 1), sigmas[index], device=device)
                    sqrt_one_minus_at = torch.full((args.n_samples, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

                    # 3. predict x0
                    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                    noise = sigma_t * noise_like(dir_xt.shape, device, args.repeat_noise) * args.temperature
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    x = x_prev

                    # 5. save
                    x_samples = model.decode_first_stage(x_prev)
                    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(
                        0, 255).astype(np.uint8)
                    x_samples = np.squeeze(x_samples)
                    img = Image.fromarray(x_samples.astype(np.uint8))
                    latent_dir = os.path.join(result_path, f'test_latent_{ts_next.item()}.jpg')
                    img.save(latent_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--precision", type=str, default="autocast")
    # 2
    parser.add_argument( "--config",type=str,default="configs/v1.yaml",)
    parser.add_argument("--ckpt",type=str,default="checkpoints/model.ckpt")
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--plms", action='store_true')
    # 3
    parser.add_argument("--outdir",type=str,default="results")

    # step 4. inference base config
    parser.add_argument("--n_samples",type=int,default=1,)
    parser.add_argument("--n_rows", type=int, default=0, )
    # step 5.
    parser.add_argument("--image_path", type=str, default="examples/image/example_1.png")
    parser.add_argument("--reference_path", type=str, default="examples/reference/example_1.jpg")
    parser.add_argument("--mask_path", type=str, default="examples/mask/example_1.png")
    # 5.5
    parser.add_argument("--guidance_scale", type=float, default=5, )
    # 5.7 timestep
    parser.add_argument("--ddim_steps",type=int,default=50,)
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="(eta=0.0 corresponds to deterministic sampling",)
    # 5.8 init noise
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    #
    parser.add_argument("--repeat_noise", action = 'store_true')
    parser.add_argument("--temperature", type=int, default=1)

    parser.add_argument("--skip_grid",action='store_true',)
    parser.add_argument("--skip_save",action='store_true',)
    parser.add_argument("--fixed_code",action='store_false')

    parser.add_argument("--n_iter",type=int,default=2,help="sample this often",)
    parser.add_argument("--n_imgs",type=int,default=100,)
    args = parser.parse_args()
    main(args)
