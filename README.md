<p align="center">
  <img src="asset/logo.png"  height=120>
</p>


### <div align="center">👉 PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis<div> 
### <div align="center"> ICLR 2024 Spotlight <div> 

<div align="center">
  <a href="https://github.com/PixArt-alpha/PixArt-sigma/"><img src="https://img.shields.io/static/v1?label=PixArt-Sigma Code&message=Github&color=blue&logo=github-pages"></a> &ensp;

  <a href="https://pixart-alpha.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M"><img src="https://img.shields.io/static/v1?label=SAM-LLaVA&message=HF&color=yellow"></a> &ensp;
  <a href="https://arxiv.org/abs/2310.00426"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Alpha&color=red&logo=arxiv"></a> &ensp;
  <a href="https://arxiv.org/abs/2401.05252"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Delta&color=red&logo=arxiv"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
  <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart"><img src="https://img.shields.io/static/v1?label=Usage&message=Diffusers&color=green&"></a> &ensp;
  <a href="https://github.com/city96/ComfyUI_ExtraModels"><img src="https://img.shields.io/static/v1?label=App&message=ComfyUI&&color=green"></a> &ensp;

  <a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha"><img src="https://img.shields.io/static/v1?label=Demo PixArt&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-LCM"><img src="https://img.shields.io/static/v1?label=Demo PixArt-LCM&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://openxlab.org.cn/apps/detail/PixArt-alpha/PixArt-alpha"><img src="https://img.shields.io/static/v1?label=Demo PixArt&message=OpenXLab&color=purple"></a> &ensp;
  <a href="https://openxlab.org.cn/apps/detail/houshaowei/PixArt-LCM"><img src="https://img.shields.io/static/v1?label=Demo PixArt-LCM&message=OpenXLab&color=purple"></a> &ensp;
  <a href="https://colab.research.google.com/drive/1jZ5UZXk7tcpTfVwnX33dDuefNMcnW9ME?usp=sharing"><img src="https://img.shields.io/static/v1?label=Free%20Trial&message=Google%20Colab&logo=google&color=orange"></a> &ensp;
</div>

---

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for pixel-art and also contains our dream diffuser code in the scirpt folder.

---

## Contents
* [Training](#-how-to-train)
* [Inference](#-how-to-test)
* [Download Models](#-download-models)
* [Use diffusers](#1---using-in--diffusers)
* [Data Processing](#-how-to-extract-t5-and-vae-features)
* [PixArt-**α** Demo](#3---gradio-with-diffusers--faster-)
* [PixArt-**α** 8GB VRAM](asset/docs/pixart.md)
* [Acknowledgement](#acknowledgements)
* [Citation](#bibtex)


# 🔧 Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n pixart python=3.9
conda activate pixart
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/PixArt-alpha/PixArt-alpha.git
cd PixArt-alpha
pip install -r requirements.txt

# Using hook to get attention map
pip install bytecode
python setup.py install

# For visualize attention map
pip install visualize==0.5.0
```

# ⏬ Download Models
All models will be automatically downloaded. You can also choose to download manually from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha).

| Model                       | #Params | url                                                                                                                                                                                                          | Download in OpenXLab                                                                                            |
|:----------------------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| T5                          | 4.3B    | [T5](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl)                                                                                                                                 | [T5](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/t5-v1_1-xxl.zip)                  |
| VAE                         | 80M     | [VAE](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema)                                                                                                                              | [VAE](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/sd-vae-ft-ema.zip)               |
| PixArt-α-SAM-256            | 0.6B    | [PixArt-XL-2-SAM-256x256.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth) or [diffusers version](https://huggingface.co/PixArt-alpha/PixArt-XL-2-SAM-256x256) | [256-SAM](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-SAM-256x256.pth) |
| PixArt-α-256                | 0.6B    | [PixArt-XL-2-256x256.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth) or [diffusers version](https://huggingface.co/PixArt-alpha/PixArt-XL-2-256x256)             | [256](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-256x256.pth)         |
| PixArt-α-256-MSCOCO-FID7.32 | 0.6B    | [PixArt-XL-2-256x256.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256-MSCOCO-FID732.pth)                                                                               | [256]()                                                                                                         |
| PixArt-α-512                | 0.6B    | [PixArt-XL-2-512x512.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth) or [diffusers version](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512)             | [512](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-512x512.pth)         |
| PixArt-α-1024               | 0.6B    | [PixArt-XL-2-1024-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth) or [diffusers version](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)             | [1024](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-1024-MS.pth)        |
| PixArt-δ-1024-LCM           | 0.6B    | [diffusers version](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS)                                                                                                                             |                                                                                                                 |
| PixArt-δ-512-ControlNet     | 0.9B    | [PixArt-XL-2-512-ControlNet.pth](https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main)                                                                                                            | [512](https://openxlab.org.cn/models/detail/PixArt-alpha/PixArt-ControlNet)                                     |
| PixArt-δ-1024-ControlNet    | 0.9B    | [PixArt-XL-2-1024-ControlNet.pth](https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main)                                                                                                           | [1024](https://openxlab.org.cn/models/detail/PixArt-alpha/PixArt-ControlNet)                                    |

ALSO find all models in [OpenXLab_PixArt-alpha](https://openxlab.org.cn/models/detail/PixArt-alpha/PixArt-alpha)

# 🔥 How to Train
## 1. PixArt Training

**First of all.**

Thanks to [@kopyl](https://github.com/kopyl), you can reproduce the full fine-tune training flow on [Pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) from HugginFace with notebooks:
1. Train with [notebooks/train.ipynb](https://github.com/PixArt-alpha/PixArt-alpha/blob/53dac066f60fe5fdbdde4f0360145ca96d4cc38c/notebooks/train.ipynb).
2. Convert to Diffusers with [notebooks/convert-checkpoint-to-diffusers.ipynb](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/notebooks/convert-checkpoint-to-diffusers.ipynb).
3. Run the inference with converted checkpoint in step 2 with [notebooks/infer.ipynb](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/notebooks/infer.ipynb).

**Then, for more details.**

Here we take SAM dataset training config as an example, but of course, you can also prepare your own dataset following this method.

You **ONLY** need to change the **config** file in [config](./configs/pixart_config) and **dataloader** in [dataset](./diffusion/data/datasets).
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train_scripts/train.py configs/pixart_config/PixArt_xl2_img256_SAM.py --work-dir output/train_SAM_256
```

The directory structure for SAM dataset is:
```
cd ./data

SA1B
├──images/  (images are saved here)
│  ├──sa_xxxxx.jpg
│  ├──sa_xxxxx.jpg
│  ├──......
├──captions/    (corresponding captions are saved here, same name as images)
│  ├──sa_xxxxx.txt
│  ├──sa_xxxxx.txt
├──partition/   (all image names are stored txt file where each line is a image name)
│  ├──part0.txt
│  ├──part1.txt
│  ├──......
├──caption_feature_wmask/   (run tools/extract_caption_feature.py to generate caption T5 features, same name as images except .npz extension)
│  ├──sa_xxxxx.npz
│  ├──sa_xxxxx.npz
│  ├──......
├──img_vae_feature/  (run tools/extract_img_vae_feature.py to generate image VAE features, same name as images except .npy extension)
│  ├──train_vae_256/
│  │  ├──noflip/
│  │  │  ├──sa_xxxxx.npy
│  │  │  ├──sa_xxxxx.npy
│  │  │  ├──......

```

**Here we prepare data_toy for better understanding**
```bash
cd ./data

git lfs install
git clone https://huggingface.co/datasets/PixArt-alpha/data_toy
```
Then, 
[Here](https://huggingface.co/datasets/PixArt-alpha/data_toy/blob/main/part0.txt) is an example of partition/part0.txt file.

---

Besides, for json file guided [training](https://github.com/PixArt-alpha/PixArt-alpha/blob/fe0cb78065d64c18ecd8955a04e4f29138d47946/configs/pixart_config/PixArt_xl2_img1024_internalms.py#L3C2-L3C2),
[here](https://huggingface.co/datasets/PixArt-alpha/data_toy/blob/main/data_info.json) is a toy json file for better understand.

---

## 2. PixArt + DreamBooth Training

Following the `Pixart + DreamBooth` [training guidance](asset/docs/pixart-dreambooth.md)

## 3. PixArt + LCM / LCM-LoRA Training

Following the `PixArt + LCM` [training guidance](asset/docs/pixart_lcm.md)

## 4. PixArt + ControlNet Training

Following the `PixArt + ControlNet` [training guidance](asset/docs/pixart_controlnet.md)

## 4. PixArt + LoRA Training

```bash
pip install peft==0.6.2

accelerate launch --num_processes=1 --main_process_port=36667  train_scripts/train_pixart_lora_hf.py --mixed_precision="fp16" \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
  --dataset_name=lambdalabs/pokemon-blip-captions --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=200 --checkpointing_steps=100 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="pixart-pokemon-model" \
  --validation_prompt="cute dragon creature" --report_to="tensorboard" \
  --gradient_checkpointing --checkpoints_total_limit=10 --validation_epochs=5 \
  --rank=16
```

# 💻 How to Test
Inference requires at least `23GB` of GPU memory using this repo, while `11GB and 8GB` using in 🧨 [diffusers](#using-in--diffusers).

Currently support:
- [x] [IDDPM](https://arxiv.org/abs/2102.09672)
- [x] [DPM-Solver](https://arxiv.org/abs/2206.00927)
- [x] [SA-Solver](https://arxiv.org/abs/2309.05019)
- [ ] [DPM-Solver-v3](https://arxiv.org/abs/2310.13268v2)

## 1. Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

To get started, first install the required dependencies. Make sure you've downloaded the [models](https://huggingface.co/PixArt-alpha/PixArt-alpha) to the output/pretrained_models folder, and then run on your local machine:

```bash
DEMO_PORT=12345 python app/app.py
```

As an alternative, a sample [Dockerfile](Dockerfile) is provided to make a runtime container that starts the Gradio app.

```bash
docker build . -t pixart
docker run --gpus all -it -p 12345:12345 -v <path_to_huggingface_cache>:/root/.cache/huggingface pixart
```

Or use docker-compose.  Note, if you want to change context from the 1024 to 512 or LCM version of the app just change the APP_CONTEXT env variable in the docker-compose.yml file.  The default is 1024

```bash
docker compose build
docker compose up
```

Let's have a look at a simple example using the `http://your-server-ip:12345`.


## 2. Integration in diffusers
### 1). Using in 🧨 diffusers

Make sure you have the updated versions of the following libraries:

```bash
pip install -U transformers accelerate diffusers SentencePiece ftfy beautifulsoup4
```

And then:

```python
import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)

# If use DALL-E 3 Consistency Decoder
# pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

# If use SA-Solver sampler
# from diffusion.sa_solver_diffusers import SASolverScheduler
# pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')

# If loading a LoRA model
# transformer = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")
# pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
# del transformer

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

pipe.to(device)

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
image.save("./catcus.png")
```
Check out the [documentation](./asset/docs/sasolver.md) for more information about SA-Solver Sampler.

This integration allows running the pipeline with a batch size of 4 under 11 GBs of GPU VRAM. 
Check out the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart) to learn more.

### 2). Running the `PixArtAlphaPipeline` in under 8GB GPU VRAM

GPU VRAM consumption under 8 GB is supported now, please refer to [documentation](asset/docs/pixart.md) for more information.

### 3). Gradio with diffusers (Faster)

To get started, first install the required dependencies, then run on your local machine:

```bash
# diffusers version
DEMO_PORT=12345 python app/app.py
```
Let's have a look at a simple example using the `http://your-server-ip:12345`.

You can also click [here](https://colab.research.google.com/drive/1jZ5UZXk7tcpTfVwnX33dDuefNMcnW9ME?usp=sharing) to have a free trial on Google Colab.

### 4). Convert .pth checkpoint into diffusers version

```bash
python tools/convert_pixart_alpha_to_diffusers.py --image_size your_img_size --multi_scale_train (True if you use PixArtMS else False) --orig_ckpt_path path/to/pth --dump_path path/to/diffusers --only_transformer=True
```

# ✏️ How to extract T5 and VAE features

Prepare T5 text feature and VAE image feature in advance will speed up the training process and save GPU memory.
```bash
python tools/extract_features.py --img_size=1024 \
    --json_path "data/data_info.json" \
    --t5_save_root "data/SA1B/caption_feature_wmask" \
    --vae_save_root "data/SA1B/img_vae_features" \
    --pretrained_models_dir "output/pretrained_models" \
    --dataset_root "data/SA1B/Images/"
```


# 📖BibTeX
    @misc{chen2023pixartalpha,
          title={PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis}, 
          author={Junsong Chen and Jincheng Yu and Chongjian Ge and Lewei Yao and Enze Xie and Yue Wu and Zhongdao Wang and James Kwok and Ping Luo and Huchuan Lu and Zhenguo Li},
          year={2023},
          eprint={2310.00426},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    @misc{chen2024pixartdelta,
          title={PIXART-{\delta}: Fast and Controllable Image Generation with Latent Consistency Models}, 
          author={Junsong Chen and Yue Wu and Simian Luo and Enze Xie and Sayak Paul and Ping Luo and Hang Zhao and Zhenguo Li},
          year={2024},
          eprint={2401.05252},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
# 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for their wonderful technical support and awesome collaboration!
- Thanks to [Hugging Face](https://github.com/huggingface) for sponsoring the nicely demo!
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their wonderful work and codebase!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PixArt-alpha/PixArt-alpha&type=Date)](https://star-history.com/#PixArt-alpha/PixArt-alpha&Date)
