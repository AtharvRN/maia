# Colab OpenMAIA Real-Neuron Demo

This fork includes a Colab-ready notebook for the real-neuron MAIA pipeline:

`demo_colab_real_neuron.ipynb`

The notebook is intentionally close to the original `demo.ipynb`, but it is set
up for a single Colab A100 with OpenAI as the language agent and local FLUX
image tools.

## What Changed

- GroundingDINO/SAM is optional at import time. Real-neuron runs no longer fail
  just because synthetic-neuron dependencies are absent.
- The OpenAI adapter uses the selected `--agent` model instead of hardcoding
  `gpt-4o`.
- OpenAI message normalization works with the current `normalize_messages`
  signature.
- `gpt-5*` and `o*` models use `max_completion_tokens`; older chat models use
  `max_tokens`.
- `main.py` accepts `--max_output_tokens`, `--max_rounds`,
  `--text2image_device`, and `--img2img_device`.
- FLUX and FLUX-Kontext default to generation batch size 1, which is safer for
  Colab and memory-constrained demos.
- `clip-RN50` no longer uses a hardcoded CLIP cache path.

## Recommended Colab Path

1. Open `demo_colab_real_neuron.ipynb` in Colab.
2. Use an A100 runtime.
3. Run the install cell.
4. If Colab warns about binary package ABI changes, restart the runtime once.
5. Continue from the clone cell.
6. Set `OPENAI_API_KEY` and optionally `HF_TOKEN`.
7. Run the real-neuron demo.

Default demo target:

```text
model = clip-RN50
layer = layer4
unit = 1673
agent = gpt-4o
```

If the OpenAI TPM limit is tight, lower `max_output_tokens` or switch
`agent_name` in the notebook to `gpt-4o-mini`.

## CLI Example

For a two-GPU pod where FLUX and FLUX-Kontext should live on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --agent gpt-4o \
  --model clip-RN50 \
  --unit_mode manual \
  --units layer4=1673 \
  --path2prompts ./prompts/open \
  --path2exemplars ./exemplars \
  --path2save ./results_original_flux_split \
  --device 0 \
  --text2image_device cuda:0 \
  --img2img_device cuda:1 \
  --max_output_tokens 1024 \
  --debug
```

For a lightweight smoke test without image-generation models:

```bash
python main.py \
  --agent gpt-4o \
  --model resnet50 \
  --unit_mode manual \
  --units layer4=1673 \
  --exemplar_source cifar100 \
  --probe_size 200 \
  --exemplar_batch_size 64 \
  --path2prompts ./prompts/cifar \
  --skip_image_models \
  --device 0 \
  --debug
```

## One-A100 Kubernetes Pod

For a pod with 1x A100 80 GB VRAM and only about 32 GB host RAM, do not use the
full `install.sh`. That script also installs the synthetic-neuron stack and
GroundingDINO/SAM, which is unnecessary for the real-neuron demo.

Use the minimal setup:

```bash
git clone https://github.com/AtharvRN/maia.git
cd maia
bash scripts/setup_one_a100_pod.sh
conda activate maia-one-a100
```

Then set credentials and run:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...  # if needed for FLUX access
bash scripts/run_one_a100_real.sh
```

This runner passes `--disable_cpu_offload`, so FLUX and FLUX-Kontext stay on the
A100 instead of moving model blocks through host RAM. That is the better default
for a low-RAM pod. If VRAM becomes the bottleneck, use `--skip_image_models` for
a dataset-exemplar-only smoke test.
