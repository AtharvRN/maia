# OpenMAIA Real-Neuron Demo on a Kubernetes Pod

This fork keeps the original MAIA real-neuron flow, with a minimal setup path
for a Kubernetes pod.

Tested hardware: 2x NVIDIA A100 80 GB GPUs and 32 GB system RAM. The intended
layout is GPU 0 for the local VLM server and GPU 1 for MAIA tools.

## Setup

```bash
git clone https://github.com/AtharvRN/maia.git
cd maia

bash scripts/setup_pod_env.sh
conda activate maia-pod
```

Set credentials:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...  # if needed for FLUX access
```

## Run

```bash
bash scripts/run_pod_real.sh \
  --agent gpt-4o \
  --unit 1673
```

The script defaults to:

```text
MODEL=clip-RN50
LAYER=layer4
UNIT=1673
RESULTS_DIR=./results_pod_real
```

Use a separate results directory for reruns:

```bash
bash scripts/run_pod_real.sh \
  --agent gpt-4o \
  --unit 1673 \
  --results-dir ./results_pod_real_rerun
```

## Pod Memory Behavior

The runner passes `--disable_cpu_offload`, so FLUX and FLUX-Kontext stay on the
GPU instead of moving large model blocks through host RAM. This is the intended
mode for low-RAM pods.

## Outputs

For each unit, results are saved under:

```text
<RESULTS_DIR>/<agent>/<model>/<layer>/<unit>/
```

Important files:

- `experiment.html`: visual experiment log with displayed images embedded.
- `history.json`: full MAIA conversation, including displayed images as base64.
- `description.txt`: final description, if the agent completed the run.
- `label.txt`: final label, if provided by the agent.
