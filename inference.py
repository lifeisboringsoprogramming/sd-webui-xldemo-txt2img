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

    if seed == -1:
        seed = int(random.randrange(4294967294))

    device = 'cuda'
    generator = torch.Generator(device=device)

    generator = generator.manual_seed(seed)

    latents = torch.randn(
        (1, pipe.unet.in_channels, 1024 // 8, 1024 // 8),
        generator=generator,
        device=device,
        dtype=torch.float16
    )

    prompt = '✨aesthetic✨ aliens walk among us in Las Vegas, scratchy found film photograph'
    negative_prompt = 'low quality'
    guidance_scale = 7
    num_inference_steps = 20

    start_time = time_.time()

    images = pipe(prompt=[prompt], negative_prompt=[negative_prompt],
                  guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                  latents=latents).images

    end_time = time_.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    gc.collect()
    torch.cuda.empty_cache()

    images[0].save(f'inference-{int(time_.time())}-{seed}.png')


if __name__ == "__main__":

    # Run your code
    inference(-1)

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
    # 
    # Python 3.10.9
    # torch                   2.0.1+cu118
    # transformers            4.25.1
    # diffusers               0.18.1
    #
    # ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-0.9/snapshots/025709258a55cc924dc47efd88959f18ae79830e$ tree
    # .
    # ├── model_index.json -> ../../blobs/4b76d56998e4e2c7bfe973ccb4d0f1c361e1287b
    # ├── scheduler
    # │   └── scheduler_config.json -> ../../../blobs/5bdb7b6e0eeda414c9c37ec916da0fc4ef294c7e
    # ├── text_encoder
    # │   ├── config.json -> ../../../blobs/15cf93d7088b7f349e6522a8692c457d8ae6fde9
    # │   └── model.safetensors -> ../../../blobs/22928c6a6a99759e4a19648ba56e044d1df47b650f7879470501b71ec996a3ef
    # ├── text_encoder_2
    # │   ├── config.json -> ../../../blobs/c4ad7f842f557f4371e748443299a3c70a5dcbe1
    # │   └── model.safetensors -> ../../../blobs/d65d20651dd313f3b699b03885da0032d8f852b8b5dbbbdf5b56ce9b10ca5e3d
    # ├── tokenizer
    # │   ├── merges.txt -> ../../../blobs/76e821f1b6f0a9709293c3b6b51ed90980b3166b
    # │   ├── special_tokens_map.json -> ../../../blobs/2c2130b544c0c5a72d5d00da071ba130a9800fb2
    # │   ├── tokenizer_config.json -> ../../../blobs/2e8612a429492973fe60635b3f44a28b065cfac0
    # │   └── vocab.json -> ../../../blobs/469be27c5c010538f845f518c4f5e8574c78f7c8
    # ├── tokenizer_2
    # │   ├── merges.txt -> ../../../blobs/76e821f1b6f0a9709293c3b6b51ed90980b3166b
    # │   ├── special_tokens_map.json -> ../../../blobs/ae0c5be6f35217e51c4c000fd325d8de0294e99c
    # │   ├── tokenizer_config.json -> ../../../blobs/a8438e020c4497a429240d6b89e0bf9a6e2ffa92
    # │   └── vocab.json -> ../../../blobs/469be27c5c010538f845f518c4f5e8574c78f7c8
    # ├── unet
    # │   ├── config.json -> ../../../blobs/e53796e5812b975c00aefbeb475cce337c88fde9
    # │   └── diffusion_pytorch_model.safetensors -> ../../../blobs/7a516d65c0f41e82e7f3c16cad90d2362a01533beec7309e3606d59cd682797f
    # └── vae
    #     ├── config.json -> ../../../blobs/6e9694046afd2a944dd17a2390b98773cacf2f7c
    #     └── diffusion_pytorch_model.safetensors -> ../../../blobs/1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341
    # 7 directories, 18 files
    # 
    # Elapsed time: 29.2194 seconds

    # Windows 10
    # +---------------------------------------------------------------------------------------+
    # | NVIDIA-SMI 536.40                 Driver Version: 536.40       CUDA Version: 12.2     |
    # |-----------------------------------------+----------------------+----------------------+
    # | GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    # | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    # |                                         |                      |               MIG M. |
    # |=========================================+======================+======================|
    # |   0  NVIDIA GeForce RTX 3060      WDDM  | 00000000:05:00.0  On |                  N/A |
    # |  0%   46C    P8              14W / 170W |    557MiB / 12288MiB |      8%      Default |
    # |                                         |                      |                  N/A |
    # +-----------------------------------------+----------------------+----------------------+
    #
    # Python 3.10.6
    # torch                   2.0.1+cu118
    # transformers            4.25.1
    # diffusers               0.18.1
    #
    # C:\Users\libsp\.cache\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-0.9\snapshots\025709258a55cc924dc47efd88959f18ae79830e>tree /F
    # Folder PATH listing
    # Volume serial number is 82DA-B681
    # C:.
    # │   model_index.json
    # │
    # ├───scheduler
    # │       scheduler_config.json
    # │
    # ├───text_encoder
    # │       config.json
    # │       pytorch_model.bin
    # │
    # ├───text_encoder_2
    # │       config.json
    # │       pytorch_model.bin
    # │
    # ├───tokenizer
    # │       merges.txt
    # │       special_tokens_map.json
    # │       tokenizer_config.json
    # │       vocab.json
    # │
    # ├───tokenizer_2
    # │       merges.txt
    # │       special_tokens_map.json
    # │       tokenizer_config.json
    # │       vocab.json
    # │
    # ├───unet
    # │       config.json
    # │       diffusion_pytorch_model.bin
    # │
    # └───vae
    #         config.json
    #         diffusion_pytorch_model.bin
    #
    # Elapsed time: 69.5944 seconds
