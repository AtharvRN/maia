# OpenMAIA Real-Neuron Demo on One A100 Pod

This fork keeps the original MAIA real-neuron flow, with a minimal setup path
for a Kubernetes pod with one A100 GPU.

## Setup

```bash
git clone https://github.com/AtharvRN/maia.git
cd maia
git checkout one-a100-pod-demo

bash scripts/setup_one_a100_pod.sh
conda activate maia-one-a100
```

Set credentials:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...  # if needed for FLUX access
```

## Run

```bash
bash scripts/run_one_a100_real.sh \
  --agent gpt-4o \
  --unit 1673
```

The script defaults to:

```text
MODEL=clip-RN50
LAYER=layer4
UNIT=1673
RESULTS_DIR=./results_one_a100_real
```

Use a separate results directory for reruns:

```bash
bash scripts/run_one_a100_real.sh \
  --agent gpt-4o \
  --unit 1673 \
  --results-dir ./results_one_a100_real_rerun
```

## Pod Memory Behavior

The runner passes `--disable_cpu_offload`, so FLUX and FLUX-Kontext stay on the
A100 instead of moving large model blocks through host RAM. This is the intended
mode for low-RAM pods with one 80 GB A100.

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
