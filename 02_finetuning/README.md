# Finetuning Setup Guide

This directory contains the finetuning scripts and external repositories for training genomic language models on the Lambda prophage detection task.

## Directory Structure

After setup, this directory should look like:
```
02_finetuning/
├── README.md
├── caduceus/          # Cloned Caduceus repository
├── DNABERT_2/         # Cloned DNABERT2 repository
└── scripts/           # Custom finetuning scripts (optional)
```

## Conda Environment Setup

### Option 1: Use the Lambda Base Environment

The Lambda repository provides a base environment that can be extended:

```bash
# From the lambda root directory
conda env create -f environment.yml
conda activate lambda_v2

# Install additional packages for finetuning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install wandb  # Optional: for experiment tracking
```

### Option 2: Create Separate Environments for Each Model

This approach keeps dependencies isolated for each framework.

## Setup Instructions

### 1. Clone and Setup Caduceus

[Caduceus](https://github.com/kuleshov-group/caduceus) is a bi-directional DNA language model that supports both Caduceus and HyenaDNA architectures.

```bash
cd 02_finetuning
git clone https://github.com/kuleshov-group/caduceus.git
cd caduceus
```

#### Create Caduceus Conda Environment:

```bash
# Option A: Use the provided environment file (recommended)
conda env create -f environment.yml
conda activate caduceus

# Option B: Create manually
conda create -n caduceus python=3.10
conda activate caduceus

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Caduceus dependencies
pip install -e .

# Install additional requirements
pip install mamba-ssm causal-conv1d>=1.1.0
pip install transformers datasets wandb
pip install flash-attn --no-build-isolation  # Optional: for faster attention
```

#### Verify Caduceus Installation:
```bash
python -c "from caduceus.modeling_caduceus import CaduceusForMaskedLM; print('Caduceus installed successfully')"
```

#### Caduceus Models:
- **Caduceus-Ph** - Pre-trained on phage sequences
- **Caduceus-PS** - Pre-trained on prokaryotic sequences
- **HyenaDNA** - Long-range DNA model

For more details, see the [Caduceus GitHub](https://github.com/kuleshov-group/caduceus).

---

### 2. Clone and Setup DNABERT2

[DNABERT2](https://github.com/MAGICS-LAB/DNABERT_2) is a multi-species genome foundation model with efficient tokenization.

```bash
cd 02_finetuning
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
cd DNABERT_2
```

#### Create DNABERT2 Conda Environment:

```bash
# Option A: Use requirements file
conda create -n dnabert2 python=3.9
conda activate dnabert2
pip install -r requirements.txt

# Option B: Create manually with specific versions
conda create -n dnabert2 python=3.9
conda activate dnabert2

# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers==4.29.2
pip install datasets==2.12.0
pip install accelerate==0.20.3
pip install evaluate==0.4.0
pip install scikit-learn==1.2.2
pip install pandas numpy

# Install triton for efficient kernels (Linux only)
pip install triton==2.0.0

# Install einops for tensor operations
pip install einops
```

#### Verify DNABERT2 Installation:
```bash
python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True); print('DNABERT2 installed successfully')"
```

#### DNABERT2 Models:
- **DNABERT-2-117M** - 117M parameter model (HuggingFace: `zhihan1996/DNABERT-2-117M`)

For more details, see the [DNABERT2 GitHub](https://github.com/MAGICS-LAB/DNABERT_2).

---

## Finetuning for Prophage Detection

### Data Preparation

The Lambda benchmark provides finetuning data in `data/FragmentClassification/`:
- `train.csv` - Training sequences with labels
- `dev.csv` - Validation sequences with labels
- `test.csv` - Test sequences with labels

Each CSV has columns: `sequence`, `label` (0 = bacterial, 1 = prophage)

### Finetuning with Caduceus

```bash
cd 02_finetuning/caduceus
conda activate caduceus

# Example finetuning command (adjust paths as needed)
python train.py \
    --model_name caduceus-ps \
    --train_data ../../data/FragmentClassification/train.csv \
    --val_data ../../data/FragmentClassification/dev.csv \
    --output_dir ./finetuned_models/prophage \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-5
```

### Finetuning with DNABERT2

```bash
cd 02_finetuning/DNABERT_2
conda activate dnabert2

# Example finetuning command (adjust paths as needed)
python finetune.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --train_file ../../data/FragmentClassification/train.csv \
    --validation_file ../../data/FragmentClassification/dev.csv \
    --output_dir ./finetuned_models/prophage \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1
```

---

## Model Checkpoints

After finetuning, save your model checkpoints to the main Lambda directory:
```bash
# Copy finetuned models to the checkpoint directory
mkdir -p ../../phage_finetuned_model_ckpts/MODEL_NAME
cp -r ./finetuned_models/prophage/* ../../phage_finetuned_model_ckpts/MODEL_NAME/
```

These checkpoints can then be used by the inference scripts in `03_inference/`.

---

## Environment Summary

| Environment | Python | PyTorch | Purpose |
|-------------|--------|---------|---------|
| `lambda_v2` | 3.9 | - | Preprocessing, data handling |
| `caduceus` | 3.10 | 2.0+ | Caduceus/HyenaDNA finetuning |
| `dnabert2` | 3.9 | 2.0.1 | DNABERT2 finetuning |
| `dna` | 3.9 | 2.0+ | Inference (used by wrapper scripts) |

---

## Additional Resources

- [Caduceus Paper](https://arxiv.org/abs/2403.03234)
- [DNABERT2 Paper](https://arxiv.org/abs/2306.15006)
- [HuggingFace Lambda Dataset](https://huggingface.co/datasets/leannmlindsey/lambda)

---

## Troubleshooting

### CUDA/GPU Issues
Ensure you have compatible CUDA drivers installed. Both Caduceus and DNABERT2 require GPU support for efficient training.

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check GPU memory
nvidia-smi
```

### Memory Issues
If you encounter OOM (Out of Memory) errors, try:
- Reducing batch size (`--per_device_train_batch_size 8`)
- Using gradient accumulation (`--gradient_accumulation_steps 4`)
- Enabling mixed precision training (`--fp16`)
- Using gradient checkpointing (`--gradient_checkpointing`)

### Package Conflicts
If you encounter dependency conflicts:
```bash
# Create a fresh environment
conda create -n fresh_env python=3.9
conda activate fresh_env

# Install packages one at a time to identify conflicts
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers
# ... continue with other packages
```

### Triton Issues (DNABERT2)
Triton is only supported on Linux. On macOS or Windows:
```bash
# Skip triton installation
pip install transformers datasets accelerate evaluate scikit-learn einops
```
