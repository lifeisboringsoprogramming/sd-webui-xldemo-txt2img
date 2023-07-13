from diffusers import DiffusionPipeline
import torch
import gc
import time as time_
import random


def inference(seed=-1):
    access_token = ''

    model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

    print("Loading model", model_key_base)
    pipe = DiffusionPipeline.from_pretrained(
        model_key_base, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)
    pipe.enable_model_cpu_offload()

    pipe_refiner = DiffusionPipeline.from_pretrained(
        model_key_refiner, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)
    pipe_refiner.enable_model_cpu_offload()

    if seed == -1:
        seed = int(random.randrange(4294967294))

    device = 'cuda'
    generator = torch.Generator(device=device)

    num_images = 1
    prompt = '✨aesthetic✨ aliens walk among us in Las Vegas, scratchy found film photograph'
    negative_prompt = 'low quality'
    guidance_scale = 7
    num_inference_steps = 20
    refiner_strength = 0.3

    generators = [generator.manual_seed(seed + i) for i in range(num_images)]

    images = pipe(prompt=[prompt] * num_images, negative_prompt=[negative_prompt] * num_images, guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps, generator=generators, output_type="latent").images

    print(images[0].shape)

    gc.collect()
    torch.cuda.empty_cache()

    # for i, img in enumerate(images):
    #     img.save(
    #         f'inference-{int(time_.time())}-{str(i+1).zfill(2)}-{seed+i}-latent.png')

    # generators = [generator.manual_seed(seed + i) for i in range(num_images)]

    images = pipe_refiner(prompt=[prompt] * num_images, negative_prompt=[negative_prompt] * num_images, image=images,
                          num_inference_steps=num_inference_steps, strength=refiner_strength, generator=generators).images

    gc.collect()
    torch.cuda.empty_cache()

    for i, img in enumerate(images):
        img.save(
            f'inference-{int(time_.time())}-{str(i+1).zfill(2)}-{seed+i}-refiner-latent.png')


if __name__ == "__main__":
    start_time = time_.time()

    # Run your code
    inference(2163361334)

    end_time = time_.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
