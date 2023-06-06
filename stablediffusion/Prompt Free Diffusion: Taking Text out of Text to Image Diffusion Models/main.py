import argparse
import os.path as osp
from PIL import Image
import numpy as np
import time
import torch
import torchvision.transforms as trans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from collections import OrderedDict
from lib.model_zoo.ddim import DDIMSampler
import os
from tqdm import tqdm
from lib.model_zoo.diffusion_utils import noise_like

base_dir = r'/data7/sooyeon/Prompt-Free-Diffusion/Prompt-Free-Diffusion/models--shi-labs--prompt-free-diffusion/snapshots/b9b8a9079e2457f4c5af77d7e3261e03a5747e46'
controlnet_path = OrderedDict([['canny', ('canny', os.path.join(base_dir,'pretrained/controlnet/control_sd15_canny_slimmed.safetensors'))]])
preprocess_method = ['canny',]
diffuser_path = OrderedDict([['Deliberate-v2.0',os.path.join(base_dir,'pretrained/pfd/diffuser/Deliberate-v2-0.safetensors')]])

def load_sd_from_file(target):
    if osp.splitext(target)[-1] == '.ckpt':
        sd = torch.load(target, map_location='cpu')['state_dict']
    elif osp.splitext(target)[-1] == '.pth':
        sd = torch.load(target, map_location='cpu')
    elif osp.splitext(target)[-1] == '.safetensors':
        from safetensors.torch import load_file as stload
        sd = OrderedDict(stload(target, device='cpu')) ################################
    else:
        assert False, "File type must be .ckpt or .pth or .safetensors"
    return sd


class prompt_free_diffusion(object):

    def __init__(self,fp16=False,        # True
                 tag_ctx=None,           # '--tag_ctx', default = 'SeeCoder'
                 tag_diffuser=None,      # '--tag_diffuser', default='Deliberate-v2.0'
                 tag_ctl=None,):         # '--tag_ctl', default='canny'
        self.tag_diffuser = tag_diffuser # '--tag_diffuser', default='Deliberate-v2.0'
        self.tag_ctl = tag_ctl           # '--tag_ctl', default='canny'
        self.strict_sd = True            # True
        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 1.1 get config file from the file (yaml file) (prompt free diffuser)
        cfgm = model_cfg_bank()('pfd_seecoder_with_controlnet')    # model config bank and get config of pfd_seecoder_with_controlnet
        print(f'cfgm : {cfgm}')

        # 1.2 get model with the config file
        self.net = get_model()(cfgm)                               # model from pfd_seecoder_with_controlnet configuration

        # 1.3 vae model
        sdvae = os.path.join(base_dir,'pretrained/pfd/vae/sd-v2-0-base-autokl.pth')
        sdvae = torch.load(sdvae)
        self.net.vae['image'].load_state_dict(sdvae)

        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 2. get seecoder # ctx = "c"on"t"rol"x"  # ctx model with scheduelr ...
        self.tag_ctx = tag_ctx
        self.load_ctx(tag_ctx)

        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 3. diffuser : Deliberate-v2.0 # ---------------- updated the network
        self.load_diffuser(tag_diffuser)

        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 4. canny (control_net)
        self.load_ctl(tag_ctl)

        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 5. sampler
        self.sampler = DDIMSampler(self.net)
        self.net.eval()


    # --------------------------------------------------------------------------------------------------------------- #
    def load_ctx(self, tag):
        pretrained = os.path.join(base_dir,'pretrained/pfd/seecoder/seecoder-v1-0.safetensors')
        if tag == 'SeeCoder-PA':
            from lib.model_zoo.seecoder import PPE_MLP
            pe_layer = PPE_MLP(freq_num=20, freq_max=None, out_channel=768, mlp_layer=3)
            if self.dtype == torch.float16:
                pe_layer = pe_layer.half()
            if self.use_cuda:
                pe_layer.to('cuda')
            pe_layer.eval()
            self.net.ctx['image'].qtransformer.pe_layer = pe_layer
        else:
            self.net.ctx['image'].qtransformer.pe_layer = None
        if pretrained is not None:
            sd = load_sd_from_file(pretrained)
            sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() if ki.find('ctx.') != 0]  # control related parameters
            sd.update(OrderedDict(sd_extra))
            self.net.load_state_dict(sd, strict=True)
            print('Load context encoder from [{}] strict [{}].'.format(pretrained, True))
        self.tag_ctx = tag
        return tag

    def load_diffuser(self, tag):
        pretrained = diffuser_path[tag]
        if pretrained is not None:
            sd = load_sd_from_file(pretrained)
            if len([ki for ki in sd.keys() if ki.find('diffuser.image.context_blocks.') == 0]) == 0:
                sd = [(ki.replace('diffuser.text.context_blocks.', 'diffuser.image.context_blocks.'), vi) for ki, vi in sd.items()]
                sd = OrderedDict(sd)
            sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() if ki.find('diffuser.') != 0]
            sd.update(OrderedDict(sd_extra))
            self.net.load_state_dict(sd, strict=True)
            print('Load diffuser from [{}] strict [{}].'.format(pretrained, True))
        self.tag_diffuser = tag
        return tag

    def load_ctl(self, tag):
        """ tag = tag_ctl = canny"""
        pretrained = controlnet_path[tag][1]
        if pretrained is not None:
            sd = load_sd_from_file(pretrained)
            self.net.ctl.load_state_dict(sd, strict=True)
            print('Load controlnet from [{}] strict [{}].'.format(pretrained, True))
        self.tag_ctl = tag
        return tag


def main(args) :

    # -------------------------------------------------------------------------------------------------------------------------------- #
    print(f'\nstep1. model')
    pfd_inference = prompt_free_diffusion(fp16=args.fp16,tag_ctx=args.tag_ctx,tag_diffuser=args.tag_diffuser,tag_ctl=args.tag_ctl)
    print(f' (1.1) model')
    model = pfd_inference.net  # PromptFreeDiffusion_with_control   # def ctl.preprocess
    model.to(args.device)
    print(f' (1.2) vae')
    vae = model.vae['image']
    print(f' (1.3) seecoder')
    seecoder = model.ctx['image']
    print(f'seecoder (swin transformer): {seecoder.__class__.__name__}')
    print(f' (1.4) unet(diffuser)')
    unet = model.diffuser['image']
    print(f' (1.5) control net (half unet)')
    control_net = model.ctl
    print(f'control_net : {control_net.__class__.__name__}')

    # -------------------------------------------------------------------------------------------------------------------------------- #
    print(f'\nstep2. precision')
    if args.fp16:
        model = model.half()
        seecoder.fp16 = True
        dtype = torch.float16
    else:
        dtype = torch.float32

    # -------------------------------------------------------------------------------------------------------------------------------- #
    print(f'\nstep3. cross attention condition (classifier free guidance)')
    print(f' (3.1.1) pixel info (seecoder encoding, instead of text encoder) ')
    batch = args.n_sample_image
    pil_ref_img = Image.open(args.ref_img)
    c_raw = trans.ToTensor()(pil_ref_img)[None].to(args.device).to(dtype)
    cond = seecoder.encode(c_raw)                                                   # like text encoder
    cond = cond.repeat(batch, 1, 1)
    
    print(f' (3.1.2) unconditional (classifier free guidance)')
    un_cond = torch.zeros_like(cond).to(args.device).to(dtype)
    c_info = {'type': 'image',
              'conditioning': cond,
              'unconditional_conditioning': un_cond,
              'unconditional_guidance_scale': args.guidance_scale,}


    # --------------------------------------------------------------------------------------------------------------------------------
    print(f'\nstep4. structure guidance (to controlnet)')
    ctl_input = args.canny_img
    ctl_input = Image.open(ctl_input)
    w = args.w // 64 * 64
    h = args.h // 64 * 64
    ctl_input = ctl_input.resize([w, h], Image.Resampling.BICUBIC)
    ccraw = trans.ToTensor()(ctl_input)[None].to(args.device).to(dtype)
    cc = control_net.preprocess(ccraw, type='canny', size=[h, w])
    cc = cc.to(dtype) # shape = [Batch,3, 512,512]
    c_info['control'] = cc

    print(f'\nstep4. init latent')
    image_latent_dim = 4  # following unet config
    shape = [batch, image_latent_dim, h // 8, w // 8]
    seed = args.seed
    torch.manual_seed(seed)
    x_info = {'type': 'image', }
    x_info['x'] = torch.randn(shape, device=args.device, dtype=dtype)

    print(f'\nstep5. inference timestep and time setting')
    sampler = pfd_inference.sampler
    sampler.make_schedule(ddim_num_steps=args.ddim_steps,ddim_eta=0.0,verbose=True)
    timesteps = sampler.ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    sigmas = sampler.ddim_sigmas


    print(f'\nstep6. other inference arguments')
    temperature = 1
    noise_dropout = 0
    output_dir = r'output'
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nstep7. iterable inference')
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    for i, step in enumerate(iterator):
        ts = torch.full((batch,), step, device=args.device, dtype=dtype)
        index = total_steps - i - 1

        x = x_info['x']
        unconditional_guidance_scale = c_info['unconditional_guidance_scale']
        b, *_, device = *x.shape, x.device

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([ts] * 2)
        c_in = torch.cat([c_info['unconditional_conditioning'],
                          c_info['conditioning']])
        x_info['x'] = x_in
        c_info['c'] = c_in
        print(f'x_in : {x_in.shape} | t_in : {t_in.shape} | c_in : {c_in.shape}')
        
        
        with torch.no_grad() :
            e_t_uncond, e_t = model.apply_model(x_info,t_in,c_info).chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        extended_shape = [b] + [1] * (len(e_t.shape) - 1)
        a_t = torch.full(extended_shape, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        direction_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        random_noise = sigma_t * noise_like(x, False) * args.temperature
        print(f' - paper (12) equation, current latent')
        x = a_prev.sqrt() * pred_x0 + direction_xt + random_noise
        x_info['x'] = x
        ################################################## timestep-wise image sabe ###############################################
        with torch.no_grad() :
            imout = model.vae_decode(x, which='image')
            imout = [trans.ToPILImage()(i) for i in imout][0]
            save_dir = os.path.join(output_dir, f'man_{ts.item()}.jpg')
            imout.save(save_dir)
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # 1.
    parser.add_argument('--fp16', action = 'store_false')
    parser.add_argument('--tag_ctx', default = 'SeeCoder')
    parser.add_argument('--tag_diffuser', default='Deliberate-v2.0')
    parser.add_argument('--tag_ctl', default='canny')
    parser.add_argument('--device', default='cuda:0')
    # 2.
    parser.add_argument('--ddim_steps', default=50, type=int)
    parser.add_argument('--n_sample_image', default=1, type=int)
    parser.add_argument('--ddim_eta', default=0.0, type=float)
    parser.add_argument('--image_latent_dim', default=4, type=int)
    # step2. image setting
    parser.add_argument('--ref_img', default=r'assets/examples/man_pixcel.jpg', type=str)
    parser.add_argument('--canny_img',default=r'assets/examples/man_structure.png', type=str)
    parser.add_argument('--w', default=512, type=int)
    parser.add_argument('--h', default=512, type=int)
    # init_latent
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    # step 7
    parser.add_argument('--temperature', default=1, type=float)
    args = parser.parse_args()
    main(args)
