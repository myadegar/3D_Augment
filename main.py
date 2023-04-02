from PIL import Image
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
from torchvision import transforms
import numpy as np
import torch
from torch import autocast
import math
from contextlib import nullcontext
import time
from lovely_numpy import lo
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from omegaconf import OmegaConf
import os
import mimetypes
import shutil
import argparse


class Augment_3D():

    def __init__(self, gpu_idx):
        ckpt_path = '105000.ckpt'
        config_path = 'configs/sd-objaverse-finetune-c_concat-256.yaml'

        self.device = f'cuda:{gpu_idx}'
        config = OmegaConf.load(config_path)
            
        # Instantiate all models beforehand for efficiency.
        self.models = dict()
        print('Instantiating LatentDiffusion...')
        t0 = time.time()
        self.models['turncam'] = self.load_model_from_config(config, ckpt_path, device=self.device)
        print('Instantiating Carvekit HiInterface...')
        t1 = time.time()
        self.models['carvekit'] = create_carvekit_interface()
        t2 = time.time()
        print(f'LatentDiffusion load time : {round(t1 - t0, 1)}')
        print(f'Carvekit HiInterface load time : {round(t2 - t1, 1)}')
       

    def load_model_from_config(self, config, ckpt, device, verbose=False):
        print(f'Loading model from {ckpt}')
        pl_sd = torch.load(ckpt, map_location='cpu')
        if 'global_step' in pl_sd:
            print(f'Global Step: {pl_sd["global_step"]}')
        sd = pl_sd['state_dict']
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print('missing keys:')
            print(m)
        if len(u) > 0 and verbose:
            print('unexpected keys:')
            print(u)

        model.to(device)
        model.eval()
        return model


    @torch.no_grad()
    def sample_model(self, input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                    ddim_eta, x, y, z):
        precision_scope = autocast if precision == 'autocast' else nullcontext
        with precision_scope('cuda'):
            with model.ema_scope():
                c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
                T = torch.tensor([math.radians(x), math.sin(
                    math.radians(y)), math.cos(math.radians(y)), z])
                T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
                c = torch.cat([c, T], dim=-1)
                c = model.cc_projection(c)
                cond = {}
                cond['c_crossattn'] = [c]
                c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
                cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                    .repeat(n_samples, 1, 1, 1)]
                if scale != 1.0:
                    uc = {}
                    uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                else:
                    uc = None

                shape = [4, h // 8, w // 8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=None)
                print(samples_ddim.shape)
                # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()



    def preprocess_image(self, input_im, preprocess, size):
        '''
        :param input_im (PIL Image).
        :return input_im (H, W, 3) array in [0, 1].
        '''

        print('old input_im:', input_im.size)
        start_time = time.time()

        if preprocess:
            input_im = load_and_preprocess(self.models['carvekit'], input_im, size)
            input_im = (input_im / 255.0).astype(np.float32)
            # (H, W, 3) array in [0, 1].
        else:
            input_im = input_im.resize(size, Image.Resampling.LANCZOS)
            input_im = np.asarray(input_im, dtype=np.float32) / 255.0
            # (H, W, 4) array in [0, 1].

            # old method: thresholding background, very important
            # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

            # new method: apply correct method of compositing to avoid sudden transitions / thresholding
            # (smoothly transition foreground to white background based on alpha values)
            alpha = input_im[:, :, 3:4]
            white_im = np.ones_like(input_im)
            input_im = alpha * input_im + (1.0 - alpha) * white_im

            input_im = input_im[:, :, 0:3]
            # (H, W, 3) array in [0, 1].

        print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        print('new input_im:', lo(input_im))

        return input_im



    def main_run(self, im_path, x_list, y_list, z_list,
                preprocess=True, scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
                precision='fp32', h=256, w=256):
        '''
        :param raw_im (PIL Image).
        '''
        all_generated_images = []
        raw_im = Image.open(im_path).convert('RGBA')
        
        raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
        input_im = self.preprocess_image(raw_im, preprocess, size=[w, h])

        show_in_im = (input_im * 255.0).astype(np.uint8)
        preproc_image = Image.fromarray(show_in_im)

        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(self.device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(self.models['turncam'])
        idx = 0
        for x, y, z in zip(x_list, y_list, z_list):
            idx += 1
            print(f'\nGenerate sample {idx}:')
            x_samples_ddim = self.sample_model(input_im, self.models['turncam'], sampler, precision, h, w,
                                            ddim_steps, n_samples, scale, ddim_eta, x, y, z)
            output_imgs = []
            for x_sample in x_samples_ddim:
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                output_imgs.append(Image.fromarray(x_sample.astype(np.uint8)))
            all_generated_images.append(output_imgs)

        return preproc_image, all_generated_images
    


    def __call__(self, input_dir_path, output_dir_path, polar_range, azimuth_range, radius_range,
                 preprocess=True, scale=3.0, sample_per_image=30, ddim_steps=50, ddim_eta=1.0,
                precision='fp32', h=256, w=256):
        
        n_samples = 1
        try:
            shutil.rmtree(output_dir_path)
        except:
            pass
        os.makedirs(output_dir_path, exist_ok=True)
        files = os.listdir(input_dir_path)
        images_file = [file for file in files if str(mimetypes.guess_type(file)[0]).startswith('image')]
        if not images_file:
            raise Exception('There is no image file in directory !')
        images_path = [os.path.join(input_dir_path, image_file) for image_file in images_file]
        for image_path in images_path:
            image_name, _ = os.path.splitext(os.path.basename(image_path))
            print(f'\nProcess on file: {image_path}')
            x_list = []
            y_list = []
            z_list = []
            for _ in range(sample_per_image):
                x = int(np.round(np.random.uniform(polar_range[0], polar_range[1])))  
                y = int(np.round(np.random.uniform(azimuth_range[0], azimuth_range[1])))
                z = float(np.round(np.random.uniform(radius_range[0], radius_range[1]), 1))
                if abs(x) > 20 and abs(y) > 30:
                    x = int(np.round(np.random.uniform(-10, 10)))
                if y < -25 and z < -0.4:
                    z = float(np.round(np.random.uniform(0, 0.5), 1))
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
            preproc_image, generated_images = self.main_run(image_path, x_list, y_list, z_list,
                preprocess, scale, n_samples, ddim_steps, ddim_eta,
                precision, h, w)
            ##
            preproc_img_path = os.path.join(output_dir_path, image_name + '_preproc.png')
            preproc_image.save(preproc_img_path)
            result_dir = os.path.join(output_dir_path, image_name)
            os.makedirs(result_dir, exist_ok=True)
            for idx, gen_imgs in enumerate(generated_images):
                x = x_list[idx]
                y = y_list[idx]
                z = z_list[idx]
                if x < 0:
                    x = 'n' + str(-x)
                if y < 0:
                    y = 'n' + str(-y)
                if z < 0:
                    z = 'n' + str(-z)
                for i, gen_img in enumerate(gen_imgs):
                    gen_img_path = os.path.join(result_dir, f'img_{i}_{x}_{y}_{z}.png')
                    gen_img.save(gen_img_path)



if __name__ == '__main__':
    # params
    polar_range = (-40, 40)
    azimuth_range = (-45, 50)
    radius_range = (-0.8, 1.0)
    preprocess = True
    scale = 7.0
    sample_per_image = 30
    ddim_steps = 150
    ddim_eta = 1.0
    precision = 'fp32'
    h = 512
    w = 384

    input_dir_path = 'input_images'
    output_dir_path = 'results'
    gpu_idx = 1
    ########
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cloth_dir', type=str, default='./input_images',
                        help='Input directory path of cloth images.')
    parser.add_argument('-r', '--result_dir', type=str, default='./results_new', help='Output directory of result.')
    parser.add_argument('-n', '--n_sample', type=int, default=50,
                        help='Number of samples per image.')
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help='Index of gpu')
    
    args = parser.parse_args()
    
    input_dir_path = args.cloth_dir
    output_dir_path = args.result_dir
    gpu_idx = args.gpu_idx
    sample_per_image = args.n_sample
    
    ########

    augment_3d = Augment_3D(gpu_idx=gpu_idx)
    augment_3d(input_dir_path, output_dir_path, polar_range, azimuth_range, radius_range,
                 preprocess, scale, sample_per_image, ddim_steps, ddim_eta,
                precision, h, w)
                             
                             


  
                      

