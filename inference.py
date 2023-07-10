from diffusers import DiffusionPipeline
import torch
import gc
import time as time_

def inference():
    access_token = ''

    model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

    print("Loading model", model_key_base)
    pipe = DiffusionPipeline.from_pretrained(
        model_key_base, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)
    pipe.enable_model_cpu_offload()

    latents = None

    seed = 1234

    device = 'cuda'
    generator = torch.Generator(device=device)

    generator = generator.manual_seed(seed)

    latents = torch.randn(
        (1, pipe.unet.in_channels, 1024 // 8, 1024 // 8),
        generator=generator,
        device=device,
        dtype=torch.float16
    )

    prompt = 'Marble statue of a serene goddess, with flowing robes, delicate features, and a tranquil expression, highly detailed, soft lighting, grace and beauty'
    negative_prompt = 'low quality'
    guidance_scale = 7
    num_inference_steps = 20

    images = pipe(prompt=[prompt], negative_prompt=[negative_prompt],
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                latents=latents).images

    gc.collect()
    torch.cuda.empty_cache()

    images[0].save('inference.png')


if __name__ == "__main__":
    start_time = time_.time()
        
    # Run your code
    inference()

    end_time = time_.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # Ubuntu
    # Elapsed time: 32.2543 seconds