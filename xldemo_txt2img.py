import gradio as gr
import torch
import gc
import json
import random

from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler as dpmsolver_multistep
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler as deis_multistep
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler as unipc_multistep
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler as k_dpm_2_ancestral_discrete
from diffusers.schedulers.scheduling_ddim import DDIMScheduler as ddim
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler as dpmsolver_singlestep
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler as euler_ancestral_discrete
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as ddpm
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler as euler_discrete
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler as k_dpm_2_discrete
from diffusers.schedulers.scheduling_pndm import PNDMScheduler as pndm
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler as dpmsolver_sde
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler as lms_discrete
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler as heun_discrete

from modules.shared import opts
import modules.images as sd_images
from modules import generation_parameters_copypaste
from modules.devices import torch_gc, has_mps

XLDEMO_MODEL_CHOICES = ["SDXL 0.9",
                        "SDXL 0.9 (fp16)", "SDXL 1.0", "SDXL 1.0 (fp16)"]

XLDEMO_GENERATOR_DEVICE_CHOICES = ['cpu', 'cuda', "I don't know", "I don't care"]

XLDEMO_HUGGINGFACE_ACCESS_TOKEN = opts.data.get(
    "xldemo_txt2img_huggingface_access_token", "")

XLDEMO_LOAD_REFINER_ON_STARTUP = opts.data.get(
    "xldemo_txt2img_load_refiner_on_startup", True)

XLDEMO_MODEL = opts.data.get(
    "xldemo_txt2img_model", XLDEMO_MODEL_CHOICES[0])

XLDEMO_GENERATOR_DEVICE = opts.data.get(
    "xldemo_txt2img_generator_device", XLDEMO_GENERATOR_DEVICE_CHOICES[0])

XLDEMO_SCHEDULER_CHOICES = [
    'euler_discrete',
    'ddim',
    'ddpm',
    'deis_multistep',
    'dpmsolver_multistep',
    'dpmsolver_sde',
    'dpmsolver_singlestep',
    'euler_ancestral_discrete',
    'heun_discrete',
    'k_dpm_2_ancestral_discrete',
    'k_dpm_2_discrete',
    'lms_discrete',
    'pndm',
    'unipc_multistep',
]


def create_infotext(prompt, negative_prompt, seeds, sampler, steps, width, height, cfg_scale, index):

    generation_params = {
        "Sampler": sampler,
        "Steps": steps,
        "CFG scale": cfg_scale,
        "Seed": seeds[index],
        "Size": f"{width}x{height}",
    }

    generation_params['Model'] = XLDEMO_MODEL
    generation_params['Comment'] = "https://bit.ly/3pJKuhx"

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt[index]}" if negative_prompt else ""

    return f"{prompt[index]}{negative_prompt_text}\n{generation_params_text}".strip()


def create_infotext_for_refiner(prompt, negative_prompt, seeds, sampler, steps, width, height, index, refiner_strength):

    generation_params = {
        "Sampler": sampler,
        "Seed": seeds[index],
        "Size": f"{width}x{height}",
        "Refiner Steps": steps,
        "Refiner Strength": refiner_strength,
    }

    generation_params['Model'] = XLDEMO_MODEL
    generation_params['Comment'] = "https://bit.ly/3pJKuhx"

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt[index]}" if negative_prompt else ""

    return f"{prompt[index]}{negative_prompt_text}\n{generation_params_text}".strip()


class XLDemo:

    def __init__(self):

        self.model_name = XLDEMO_MODEL
        print(f"Using {self.model_name}")

        self.model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
        self.model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

        if self.model_name == "SDXL 1.0" or self.model_name == "SDXL 1.0 (fp16)":
            self.model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
            self.model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

        # Use refiner (eabled by default)
        self.load_refiner_on_startup = XLDEMO_LOAD_REFINER_ON_STARTUP

        try:
            if XLDEMO_HUGGINGFACE_ACCESS_TOKEN is not None and XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() != '':
                access_token = XLDEMO_HUGGINGFACE_ACCESS_TOKEN

                print("Loading model", self.model_key_base)
                self.pipe = None
                if self.model_name == 'SDXL 0.9 (fp16)' or self.model_name == 'SDXL 1.0 (fp16)':
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_key_base, torch_dtype=torch.float16, resume_download=True, variant='fp16', use_auth_token=access_token)
                else:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_key_base, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)

                if has_mps():
                    self.pipe = self.pipe.to("mps")
                else:
                    self.pipe.enable_model_cpu_offload()

        except Exception as ex:
            self.pipe = None
            print(str(ex))
            print(f'Problem loading {self.model_key_base} weight')

        try:
            if XLDEMO_HUGGINGFACE_ACCESS_TOKEN is not None and XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() != '':
                access_token = XLDEMO_HUGGINGFACE_ACCESS_TOKEN

                if self.load_refiner_on_startup:
                    print("Loading model", self.model_key_refiner)
                    self.pipe_refiner = None
                    if self.model_name == 'SDXL 0.9 (fp16)' or self.model_name == 'SDXL 1.0 (fp16)':
                        self.pipe_refiner = DiffusionPipeline.from_pretrained(
                            self.model_key_refiner, torch_dtype=torch.float16, resume_download=True, variant='fp16', use_auth_token=access_token)
                    else:
                        self.pipe_refiner = DiffusionPipeline.from_pretrained(
                            self.model_key_refiner, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)

                    if has_mps():
                        self.pipe_refiner = self.pipe_refiner.to("mps")
                    else:
                        self.pipe_refiner.enable_model_cpu_offload()

        except Exception as ex:
            self.pipe_refiner = None
            print(str(ex))
            print(f'Problem loading {self.model_key_refiner} weight')

    def get_scheduler_by_name(self, name, pipe, seeds):
        if name == 'dpmsolver_multistep':
            return dpmsolver_multistep.from_config(pipe.scheduler.config)
        elif name == 'deis_multistep':
            return deis_multistep.from_config(pipe.scheduler.config)
        elif name == 'unipc_multistep':
            return unipc_multistep.from_config(pipe.scheduler.config)
        elif name == 'k_dpm_2_ancestral_discrete':
            return k_dpm_2_ancestral_discrete.from_config(pipe.scheduler.config)
        elif name == 'ddim':
            return ddim.from_config(pipe.scheduler.config)
        elif name == 'dpmsolver_singlestep':
            return dpmsolver_singlestep.from_config(pipe.scheduler.config)
        elif name == 'euler_ancestral_discrete':
            return euler_ancestral_discrete.from_config(pipe.scheduler.config)
        elif name == 'ddpm':
            return ddpm.from_config(pipe.scheduler.config)
        elif name == 'euler_discrete':
            return euler_discrete.from_config(pipe.scheduler.config)
        elif name == 'k_dpm_2_discrete':
            return k_dpm_2_discrete.from_config(pipe.scheduler.config)
        elif name == 'pndm':
            return pndm.from_config(pipe.scheduler.config)
        elif name == 'dpmsolver_sde':
            return dpmsolver_sde.from_config({**pipe.scheduler.config, 'noise_sampler_seed': seeds})
        elif name == 'lms_discrete':
            return lms_discrete.from_config(pipe.scheduler.config)
        elif name == 'heun_discrete':
            return heun_discrete.from_config(pipe.scheduler.config)
        else:
            return euler_discrete.from_config(pipe.scheduler.config)

    def get_generator(self, seed):
        device = 'cpu'
        if XLDEMO_GENERATOR_DEVICE == 'cuda':
            device = 'cuda'

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def get_fixed_seed(self, seed):
        if seed is None or seed == '' or seed == -1:
            return int(random.randrange(4294967294))

        return seed

    def infer(self, prompt, negative, width, height, cfg_scale, seed, samples, sampler, steps):
        prompt, negative = [prompt] * samples, [negative] * samples

        images = []
        seeds = []
        gen_info_seeds = []
        images_b64_list = []
        info_texts = []

        if self.pipe:
            seed_base = self.get_fixed_seed(int(seed))
            seeds = [seed_base + i for i in range(samples)]

            generators = [self.get_generator(seeds[i]) for i in range(samples)]

            scheduler = self.get_scheduler_by_name(sampler, self.pipe, seeds)
            self.pipe.scheduler = scheduler
            self.pipe.scheduler.set_timesteps(steps)

            images = self.pipe(prompt=prompt, width=width, height=height, negative_prompt=negative, guidance_scale=cfg_scale,
                               num_inference_steps=steps, generator=generators).images

            gc.collect()
            torch_gc()
            torch.cuda.empty_cache()

            for i, image in enumerate(images):
                info = create_infotext(
                    prompt, negative, seeds, sampler, steps, width, height, cfg_scale, i)
                info_texts.append(info)
                sd_images.save_image(image, opts.outdir_txt2img_samples, '', seeds[i],
                                     prompt, opts.samples_format, info=info)
                images_b64_list.append(image)
                gen_info_seeds.append(seeds[i])

            return images_b64_list, json.dumps({'all_prompts': prompt, 'index_of_first_image': 0, 'all_seeds': gen_info_seeds, "infotexts": info_texts}), info_texts[0], ''

    def refine(self, prompt, negative, seed, sampler, steps, enable_refiner, image_to_refine, refiner_strength):
        prompt, negative = [prompt] * 1, [negative] * 1

        images = []
        seeds = []
        gen_info_seeds = []
        images_b64_list = []
        info_texts = []

        if self.load_refiner_on_startup and self.pipe_refiner and enable_refiner:

            # Get the width and height of the image
            width, height = image_to_refine.size

            images = [image_to_refine]

            seed_base = self.get_fixed_seed(int(seed))
            seeds = [seed_base + i for i in range(len(images))]

            generators = [self.get_generator(seeds[i])
                          for i in range(len(images))]

            scheduler = self.get_scheduler_by_name(
                sampler, self.pipe_refiner, seeds)
            self.pipe_refiner.scheduler = scheduler
            self.pipe_refiner.scheduler.set_timesteps(steps)

            images = self.pipe_refiner(prompt=prompt, negative_prompt=negative, image=images,
                                       num_inference_steps=steps, strength=refiner_strength, generator=generators).images

            gc.collect()
            torch_gc()
            torch.cuda.empty_cache()

            for i, image in enumerate(images):
                info = create_infotext_for_refiner(
                    prompt, negative, seeds, sampler, steps, width, height, i, refiner_strength)
                info_texts.append(info)
                sd_images.save_image(image, opts.outdir_txt2img_samples, '',
                                     seeds[i], prompt, opts.samples_format, info=info, suffix="-refiner")
                images_b64_list.append(image)
                gen_info_seeds.append(seeds[i])

            return images_b64_list, json.dumps({'all_prompts': prompt, 'index_of_first_image': 0, 'all_seeds': gen_info_seeds, "infotexts": info_texts}), info_texts[0], ''


xldemo_txt2img = XLDemo()


def can_infer():
    return xldemo_txt2img.pipe is not None


def can_refine():
    return xldemo_txt2img.pipe_refiner is not None


def do_xldemo_txt2img_infer(prompt, negative, width, height, scale, seed, samples, sampler, steps):

    try:
        return xldemo_txt2img.infer(prompt, negative, width, height, scale, seed, samples, sampler, steps)
    except Exception as ex:
        # Raise an Error with a custom error message
        raise gr.Error(f"Error: {str(ex)}")


def do_xldemo_txt2img_refine(prompt, negative, seed, sampler, steps, enable_refiner, image_to_refine, refiner_strength):

    if image_to_refine is None:
        raise gr.Error(f"Error: Please set the image for refiner")

    try:
        return xldemo_txt2img.refine(prompt, negative, seed, sampler, steps, enable_refiner, image_to_refine, refiner_strength)
    except Exception as ex:
        # Raise an Error with a custom error message
        raise gr.Error(f"Error: {str(ex)}")
