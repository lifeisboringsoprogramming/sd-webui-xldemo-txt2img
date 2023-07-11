import gradio as gr
from diffusers import DiffusionPipeline
import torch
import gc
import json
import random

from modules.shared import opts
import modules.images as sd_images
from modules import generation_parameters_copypaste
from modules.devices import get_optimal_device

XLDEMO_HUGGINGFACE_ACCESS_TOKEN = opts.data.get(
    "xldemo_txt2img_huggingface_access_token", "")

XLDEMO_LOAD_REFINER_ON_STARTUP = opts.data.get(
    "xldemo_txt2img_load_refiner_on_startup", True)


def create_infotext(prompt, negative_prompt, seeds, steps, width, height, cfg_scale, index):

    generation_params = {
        "Steps": steps,
        "CFG scale": cfg_scale,
        "Seed": seeds[index],
        "Size": f"{width}x{height}",
    }

    generation_params['Comment'] = "https://bit.ly/3pJKuhx"

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt[index]}" if negative_prompt else ""

    return f"{prompt[index]}{negative_prompt_text}\n{generation_params_text}".strip()


def create_infotext_for_refiner(prompt, negative_prompt, seeds, steps, width, height, index, refiner_strength):

    generation_params = {
        "Steps": steps,
        "Seed": seeds[index],
        "Size": f"{width}x{height}",
        "Refiner Strength": refiner_strength,
    }

    generation_params['Comment'] = "https://bit.ly/3pJKuhx"

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt[index]}" if negative_prompt else ""

    return f"{prompt[index]}{negative_prompt_text}\n{generation_params_text}".strip()


class XLDemo:

    def __init__(self):

        self.model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
        self.model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

        # Use refiner (eabled by default)
        self.load_refiner_on_startup = XLDEMO_LOAD_REFINER_ON_STARTUP

        if XLDEMO_HUGGINGFACE_ACCESS_TOKEN is not None and XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() != '':
            access_token = XLDEMO_HUGGINGFACE_ACCESS_TOKEN

            print("Loading model", self.model_key_base)
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_key_base, torch_dtype=torch.float16, resume_download=True, variant='fp16', use_auth_token=access_token)
            self.pipe.enable_model_cpu_offload()

            if self.load_refiner_on_startup:
                print("Loading model", self.model_key_refiner)
                self.pipe_refiner = DiffusionPipeline.from_pretrained(
                    self.model_key_refiner, torch_dtype=torch.float16, resume_download=True, variant='fp16', use_auth_token=access_token)
                self.pipe_refiner.enable_model_cpu_offload()

    def get_fixed_seed(self, seed):
        if seed is None or seed == '' or seed == -1:
            return int(random.randrange(4294967294))

        return seed

    def generate_latents(self, samples, width, height, in_channels, seed_base):
        device = get_optimal_device()
        generator = torch.Generator(device=device)

        latents = None
        seeds = []

        seed_base = self.get_fixed_seed(int(seed_base))
        for i in range(samples):
            # Get a new random seed, store it and use it as the generator state
            seed = seed_base + i
            seeds.append(seed)
            generator = generator.manual_seed(seed)

            image_latents = torch.randn(
                (1, in_channels, height // 8, width // 8),
                generator=generator,
                device=device,
                dtype=torch.float16
            )
            latents = image_latents if latents is None else torch.cat(
                (latents, image_latents))

        return latents, seeds

    def infer(self, prompt, negative, width, height, cfg_scale, seed, samples, steps):
        prompt, negative = [prompt] * samples, [negative] * samples

        images = []
        seeds = []
        gen_info_seeds = []
        images_b64_list = []
        info_texts = []

        if self.pipe:
            latents, seeds = self.generate_latents(
                samples, width, height, self.pipe.unet.in_channels, seed)
            images = self.pipe(prompt=prompt, negative_prompt=negative,
                               guidance_scale=cfg_scale, num_inference_steps=steps,
                               latents=latents).images

            gc.collect()
            torch.cuda.empty_cache()

            for i, image in enumerate(images):
                info = create_infotext(
                    prompt, negative, seeds, steps, width, height, cfg_scale, i)
                info_texts.append(info)
                sd_images.save_image(image, opts.outdir_txt2img_samples, '', seeds[i],
                                     prompt, opts.samples_format, info=info)
                images_b64_list.append(image)
                gen_info_seeds.append(seeds[i])

        return images_b64_list, json.dumps({'all_prompts': prompt, 'index_of_first_image': 0, 'all_seeds': gen_info_seeds, "infotexts": info_texts}), info_texts[0], ''

    def refine(self, prompt, negative, seed, steps, enable_refiner, image_to_refine, refiner_strength):
        prompt, negative = [prompt] * 1, [negative] * 1

        images = []
        seeds = []
        gen_info_seeds = []
        images_b64_list = []
        info_texts = []

        if self.load_refiner_on_startup and self.pipe_refiner and enable_refiner:
            _, seeds = self.generate_latents(1, 1, 1, 1, seed)

            # Get the width and height of the image
            width, height = image_to_refine.size

            images = [image_to_refine]
            images = self.pipe_refiner(prompt=prompt, negative_prompt=negative,
                                       image=images, num_inference_steps=steps, strength=refiner_strength).images

            gc.collect()
            torch.cuda.empty_cache()

            for i, image in enumerate(images):
                info = create_infotext_for_refiner(
                    prompt, negative, seeds, steps, width, height, i, refiner_strength)
                info_texts.append(info)
                sd_images.save_image(image, opts.outdir_txt2img_samples, '',
                                     seeds[i], prompt, opts.samples_format, info=info, suffix="-refiner")
                images_b64_list.append(image)
                gen_info_seeds.append(seeds[i])

            return images_b64_list, json.dumps({'all_prompts': prompt, 'index_of_first_image': 0, 'all_seeds': gen_info_seeds, "infotexts": info_texts}), info_texts[0], ''


xldemo_txt2img = XLDemo()


def do_xldemo_txt2img_infer(prompt, negative, width, height, scale, seed, samples, steps):

    try:
        return xldemo_txt2img.infer(prompt, negative, width, height, scale, seed, samples, steps)
    except Exception as ex:
        # Raise an Error with a custom error message
        raise gr.Error(f"Error: {str(ex)}")


def do_xldemo_txt2img_refine(prompt, negative, seed, steps, enable_refiner, image_to_refine, refiner_strength):

    if image_to_refine is None:
        raise gr.Error(f"Error: Please set the image for refiner")

    try:
        return xldemo_txt2img.refine(prompt, negative, seed, steps, enable_refiner, image_to_refine, refiner_strength)
    except Exception as ex:
        # Raise an Error with a custom error message
        raise gr.Error(f"Error: {str(ex)}")
