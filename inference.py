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

    # Ubuntu 20.04.4 LTS
    # +---------------------------------------------------------------------------------------+
    # | NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
    # |-----------------------------------------+----------------------+----------------------+
    # | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    # | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    # |                                         |                      |               MIG M. |
    # |=========================================+======================+======================|
    # |   0  NVIDIA GeForce RTX 3060         On | 00000000:05:00.0 Off |                  N/A |
    # |  0%   42C    P8               14W / 170W|    448MiB / 12288MiB |     41%      Default |
    # |                                         |                      |                  N/A |
    # +-----------------------------------------+----------------------+----------------------+    
    # Elapsed time: 32.2543 seconds