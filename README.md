
# A Multimodal Automated Interpretability Agent #
### ICML 2024 ###

### [Project Page](https://multimodal-interpretability.csail.mit.edu/maia) | [Arxiv](https://multimodal-interpretability.csail.mit.edu/maia)

<img align="right" width="42%" src="/docs/static/figures/maia_teaser.jpg">

[Tamar Rott Shaham](https://tamarott.github.io/)\*, [Sarah Schwettmann](https://cogconfluence.com/)\*, <br>
[Franklin Wang](https://frankxwang.github.io/), [Achyuta Rajaram](https://twitter.com/AchyutaBot), [Evan Hernandez](https://evandez.com/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
\*equal contribution <br><br>

MAIA is a system that uses neural models to automate neural model understanding tasks like feature interpretation and failure mode discovery. It equips a pre-trained vision-language model with a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. These include tools commonly used by human interpretability researchers: for synthesizing and editing inputs, computing maximally activating exemplars from real-world datasets, and summarizing and describing experimental results. Interpretability experiments proposed by MAIA compose these tools to describe and explain system behavior.

## News

- **Oct 18 2025** — Added support for open-source multimodal LLM backbones. Replaced *InstructDiffusion* with *FLUX.1-Kontext-dev* for image editing.
- **June 19 2025** — Released **MAIA 2.0**: free-form code execution, flexible outputs, and new backbone support (*Claude 3.5 Sonnet*, *GPT-4o*, *FLUX.1*).
- **Aug 14 2024** — Added **synthetic neurons** support.
- **July 3 2024** — Released **neuron labeling implementation**.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Kubernetes Pod Setup](#kubernetes-pod-setup)
- [Running MAIA (Core Usage)](#running-maia-core-usage)
- [Synthetic Neurons](#synthetic-neurons)
- [Using NetDissect](using-netdissect)
- [Evaluation](evaluation)
- [Acknowledgments](acknowledgments)

---

## Installation

### Prerequisites

You’ll need:

* Python ≥3.10 (recommended)
* Conda (recommended)
* HuggingFace + OpenAI/Anthropic API keys if using cloud models

### Load HuggingFace Token

Required for using FLUX.1, Gemma 3, and many different models.

```bash
export HF_TOKEN='your-hf-token-here'
```

Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Set API Keys

Required for using OpenAI and Anthropic models.

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

### Cloning the repo

```bash
git clone https://github.com/multimodal-interpretability/maia.git
cd maia
bash install.sh
bash download_exemplars.sh
```

### Serving your own models (optional)

To use an open-source model as the agent backbone, serve an open-source multimodal LLMs via [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#nvidia-cuda).

```bash
bash server/install_server.sh
```

---

## Quick Start

### (Optional) Serve a Local Model with vLLM (to use an open-source model as the agent backbone)

Run `server/serve_model.sh` to launch a **vLLM server** on **port 11434** (the default port used by Ollama), binding to **0.0.0.0** to make it accessible on all network interfaces.

```bash
bash server/serve_model.sh --model <model-name-or-repository> --gpus <number_of_gpus>

# Examples:
bash server/serve_model.sh --model mistral --gpus 4
# → Runs Mistral Small 3.2 (24B) on 4 GPUs.

bash server/serve_model.sh --model gemma --gpus 4
# → Runs Gemma 3 (27B) on 4 GPUs.

bash server/serve_model.sh --model Qwen/Qwen3-VL-30B-A3B-Instruct --gpus 1
# → Runs Qwen/Qwen3-VL-30B-A3B-Instruct from the Hugging Face repository on 1 GPU.
```

### Interactive Notebook

Run interactive demo experiments using **Jupyter Notebook**.

```bash
pip install notebook
jupyter notebook
```

Open `demo.ipynb` and follow the guided workflow.

Note: For synthetic neurons, follow the setup in `./synthetic-neurons-dataset/README.md`.

---

## Kubernetes Pod Setup

This is a repeatable setup for ephemeral Kubernetes pods.

Tested hardware: 2x NVIDIA A100 80 GB GPUs and 32 GB system RAM. It uses:

- GPU 0: local open-source VLM served with vLLM.
- GPU 1: MAIA tools, target model, FLUX text-to-image, and FLUX-Kontext image editing.

### 1. Clone or update the repo

```bash
cd /workspace
git clone https://github.com/AtharvRN/maia.git
cd /workspace/maia
```

If the repo already exists:

```bash
cd /workspace/maia
git pull
```

### 2. Create the MAIA tools environment

```bash
source /opt/conda/etc/profile.d/conda.sh

conda create -y -n maia python=3.10
conda activate maia

pip install --upgrade pip setuptools wheel

pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install \
  numpy pandas pillow matplotlib tqdm scipy scikit-learn scikit-image \
  openai anthropic tiktoken ipython notebook \
  diffusers transformers accelerate safetensors sentencepiece protobuf \
  bitsandbytes huggingface_hub hf_transfer einops timm ftfy regex \
  git+https://github.com/openai/CLIP.git \
  git+https://github.com/davidbau/baukit.git
```

### 3. Create the vLLM serving environment

```bash
source /opt/conda/etc/profile.d/conda.sh

conda create -y -n vllm-maia python=3.11
conda activate vllm-maia

pip install --upgrade pip setuptools wheel
pip install vllm
```

### 4. Optional Hugging Face token

Set this if you use gated models such as Gemma 3 or FLUX downloads that require authentication.

```bash
export HF_TOKEN="hf_..."
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

### 5. Start an officially supported local VLM on GPU 0

The repo supports `mistral` and `gemma` shortcuts through `server/serve_model.sh`.
`mistral` worked without gated Hugging Face access in our pod test.

```bash
cd /workspace/maia
mkdir -p logs

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/workspace/hf_home \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  CC=/usr/bin/gcc \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  PATH=/opt/conda/envs/vllm-maia/bin:$PATH \
  bash server/serve_model.sh --model mistral --gpus 1 \
  > logs/vllm_mistral_small32_gpu0_official.log 2>&1 &

echo $! > logs/vllm_mistral_small32_gpu0_official.pid
```

Watch startup:

```bash
tail -f /workspace/maia/logs/vllm_mistral_small32_gpu0_official.log
```

Smoke test the server:

```bash
curl -s http://127.0.0.1:11434/v1/models
```

### 6. Run MAIA on GPU 1

With `CUDA_VISIBLE_DEVICES=1`, physical GPU 1 becomes local `cuda:0`, so `--device 0`,
`--text2image_device cuda:0`, and `--img2img_device cuda:0` are correct.

```bash
cd /workspace/maia
source /opt/conda/etc/profile.d/conda.sh
conda activate maia

CUDA_VISIBLE_DEVICES=1 python main.py \
  --agent local-mistralai/Mistral-Small-3.2-24B \
  --base_url http://127.0.0.1:11434/v1 \
  --model clip-RN50 \
  --unit_mode manual \
  --units layer4=1673 \
  --path2prompts ./prompts/open \
  --path2exemplars ./exemplars \
  --path2save ./results_mistral_small32_flux_gpu1 \
  --device 0 \
  --text2image_device cuda:0 \
  --img2img_device cuda:0 \
  --max_rounds 8 \
  --max_output_tokens 1024 \
  --debug
```

---

## Running MAIA


### (Optional) Serve a Local Model with vLLM (to use an open-source model as the agent backbone)

Run `server/serve_model.sh` to launch a **vLLM server** on **port 11434** (the default port used by Ollama), binding to **0.0.0.0** to make it accessible on all network interfaces.

```bash
bash server/serve_model.sh --model <model-name-or-repository> --gpus <number_of_gpus>

# Examples:
bash server/serve_model.sh --model mistral --gpus 4
# → Runs Mistral Small 3.2 (24B) on 4 GPUs.

bash server/serve_model.sh --model gemma --gpus 4
# → Runs Gemma 3 (27B) on 4 GPUs.

bash server/serve_model.sh --model Qwen/Qwen3-VL-30B-A3B-Instruct --gpus 1
# → Runs Qwen/Qwen3-VL-30B-A3B-Instruct from the Hugging Face repository on 1 GPU.
```

### Run MAIA

After installation, you can launch **MAIA** to analyze model units or layers.
MAIA builds and runs interpretability experiments using your chosen **agent backbone**, **target model**, and **GPU**.

#### Basic usage

```bash
python main.py \
  --agent <agent_backbone> \
  --model <model_name> \
  --device <gpu_id> \
  --unit_mode <mode> \
  --units <layer_and_neuron_spec>
```

**Key arguments**

| Argument      | Description                                                                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--agent`      | **Agent backbone** for reasoning and interpretation. Options:<br>• `claude` *(default)*<br>• `gpt-4o`<br>• `local-<local_served_model_name>` (for models served via vLLM, e.g. `local-google/gemma-3-27b-it`) |
| `--base_url`  | *(Local models only)* URL where the vLLM model is served. Example: `http://localhost:11434/v1`                                                                                                                |
| `--model`     | Neural network to analyze (e.g., `resnet152`, `synthetic_neurons`).                                                                                                                                           |
| `--device`    | GPU ID for running MAIA’s agent tools (e.g., `0` or `1`).                                                                                                                                                     |
| `--unit_mode` | How neurons are selected:<br>• `manual`: specify indices directly<br>• `from_file`: load from JSON file                                                                                                       |
| `--units`     | Layers and neuron indices to analyze (manual mode). Example:<br>`layer3=229,288:layer4=122,210` → neurons 229, 288 in layer 3 and 122, 210 in layer 4.                                                        |

---

#### Examples

**1. Manual selection**

```bash
python main.py --agent gpt-4o --model resnet152 --device 0 \
  --unit_mode manual --units layer3=229,288:layer4=122,210
```

**2. From JSON file**

```bash
python main.py --agent claude --model resnet152 --device 0 \
  --unit_mode from_file --unit_file_path ./neuron_indices/
```

> Each JSON file in `./neuron_indices/` defines which layers and neurons to examine for a given model.

**3. Using a locally served model (via vLLM)**

```bash
python main.py --agent local-google/gemma-3-27b-it \
  --base_url http://localhost:11434/v1 \
  --model resnet152 --device 0 \
  --unit_mode manual --units layer3=229,288:layer4=122,210
```

---

> **System requirements:**
> MAIA’s agent tools require a **GPU with at least 24 GB VRAM** (e.g., an **RTX 3090**) and **at least 48 GB of system RAM** for stable multimodal inference and image-editing tasks.
---

> **Results** are saved as HTML files in the `./results/` directory.

---

### Large-Scale Experiments (Multi-GPU)

You can run large experiments in parallel across multiple GPUs **without any distributed setup**.
Use `--total_chunks` and `--chunk_id` to split your list of layers/units into parts — each process handles one part independently.

**Example: split across 4 GPUs**

```bash
# GPU 0
python main.py --agent gpt-4o --model resnet152 \
  --unit_mode from_file --unit_file_path ./neuron_indices/resnet152.json \
  --chunk_id 1 --total_chunks 2 --device 0 &

# GPU 1
python main.py --agent gpt-4o --model resnet152 \
  --unit_mode from_file --unit_file_path ./neuron_indices/resnet152.json \
  --chunk_id 2 --total_chunks 2 --device 1 &
```

Each run processes a different subset of units and saves results separately in `./results/`.

> No special distributed setup required — just launch one process per GPU.
> Re-run any failed chunk by reusing its same `--chunk_id`.

---

## Synthetic Neurons

### Run MAIA on Synthetic Neurons

```bash
python main.py --model synthetic_neurons --unit_mode manual --units mono=1,8:or=9:and=0,2,5
```

### Create Custom Synthetic Neurons

1. Choose mode & label(s) (e.g. `"Cheese OR Lemon"`)
2. Collect images for each concept (e.g. from COCO)
3. Compute activations via `SAMNeuron`
4. Save top images by activation in folder structure:
   `path2data/label` or `path2data/label1_label2`
5. Convert to MAIA format ([example notebook](synthetic-neurons-dataset/create_synthetic_neurons.ipynb))

---

## Generating Dataset Exemplars

MAIA uses pre-computed exemplars for its experiments. Thus, to run MAIA on a new model, you must first compute the exemplars netdissect. MAIA was built to use exemplar data as generated by MILAN’s implementation of netdissect. 

First, clone MILAN.

```python
git clone https://github.com/evandez/neuron-descriptions.git
```

Then, set the environment variables to control where the data is inputted/outputted. Ex:

```python
%env MILAN_DATA_DIR=./data
%env MILAN_MODELS_DIR=./models
%env MILAN_RESULTS_DIR=./results
```

Make those three directories mentioned above.

When you have a new model you want to dissect:

1. Make a folder in `models` directory
2. Name it whatever you want to call your model (in this case we'll use `resnet18syn`)
3. Inside that folder, place your saved model file, but name it `imagenet.pth` (since you'll be dissecting with ImageNet)

Now, place ImageNet in the data directory. If you have imagenet installed, you create a symbolic link to it using:

```bash
ln -s /path/to/imagenet /path/to/data/imagenet
```

If you don’t have imagenet installed, follow the download instructions [here](https://image-net.org/download.php): 

Next, add under your keys in `models.py` (located in `src/exemplars/models.py`):

```python
KEYS.RESNET18_SYNTHETIC = 'resnet18syn/imagenet'
```

Then add this in the keys dict:

```python
KEYS.RESNET18_SYNTHETIC:
    ModelConfig(
        models.resnet18,
        load_weights=True,
        layers=LAYERS.RESNET18,
    ),
```

Once that's done, you should be ready to run the compute exemplars script:

```bash
python3 -m scripts.compute_exemplars resnet18syn imagenet --device cuda
```

This will run the compute exemplars script using the ResNet18 synthetic model on the ImageNet dataset, utilizing CUDA for GPU acceleration.

Finally, move the computed exemplars to the `exemplars/` folder.


---

## Evaluation

Use the evaluation scripts to score neuron labels and compare backbone performance. Please note- this is a slightly different evaluation function than used in the original paper. The reproducibility of paper results is therefore not guaranteed. 

**Single-GPU example:**

```bash
python evaluation/eval.py \
  --agent <agent_name> \
  --labels <path_to_labels> \
  --n <num_prompts> \
  --device cuda:0
```

For a single GPU, omit `--chunk_id` and `--total_chunks`.

**Multi-GPU example:**

```bash
# GPU 0
python evaluation/eval.py \
  --agent <agent_name> \
  --labels <path_to_labels> \
  --n <num_prompts> \
  --chunk_id 1 --total_chunks 2 --device cuda:0 &

# GPU 1
python evaluation/eval.py \
  --agent <agent_name> \
  --labels <path_to_labels> \
  --n <num_prompts> \
  --chunk_id 2 --total_chunks 2 --device cuda:1 &
```

Each process handles a different subset of labels — no special distributed setup required; just run one process per GPU.

**Generate plots:**

```bash
python evaluation/plots.py
```

Edit `FAMILIES` in `plots.py` to point to your result directories.

---

## Acknowledgments

* [Josep Lopez](https://yusepp.github.io/) added compatibility with open-source multimodal LLMs as agents.
* [Christy Li](https://christykl.github.io/) and [Jake Touchet](https://www.linkedin.com/in/jake-touchet-557329297/) contributed to MAIA 2.0 release.
* [Christy Li](https://christykl.github.io/) also cleaned up synthetic neurons code for release.
