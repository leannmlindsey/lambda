# Lambda Benchmark Installation Guide

## Overview
Lambda (LAnguage Model Bacteriophage Detection Assessment) is a benchmark for testing genomic language models (gLMs) on bacterial and prophage detection tasks. This guide will help you set up the required datasets and run the inference pipeline.

## Supported Models

Lambda currently supports the following genomic language models:
- **DNABERT1** - DNA BERT model with k-mer tokenization
- **DNABERT2** - Improved DNA BERT with BPE tokenization
- **GROVER** - Genome-scale language model
- **NTv2** - Nucleotide Transformer v2
- **GenALM** - Genomic language model
- **EVO** - Evolution-based DNA model
- **ProkBERT** - Prokaryotic BERT model
- **megaDNA** - Large-scale DNA foundation model

## Repository Structure

```
lambda/
├── 01_preprocessing/      # Scripts for converting BED/FASTA to CSV
├── 02_finetuning/         # Finetuning scripts and external repos
├── 03_inference/          # Inference scripts for all models
├── 04_signal_extraction/  # Post-processing and metrics calculation
├── data/
│   ├── BED/
│   ├── CSV/
│   ├── FASTA/
│   ├── FragmentClassification/
│   └── genomes/
└── results/               # Model predictions (created at runtime)
```

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/leannmlindsey/lambda.git
cd lambda
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate lambda
```

3. Download the datasets and place them into the data directory

4. Navigate to the data directory:
```bash
cd data
```

5. Extract the archive files:
```bash
tar -xf BED.tar
tar -xf FASTA.tar
tar -xf FragmentClassification.tar
```

After extraction, your directory structure should look like this:
```
data/
├── BED/
├── FASTA/
└── FragmentClassification/
```

6. Extract overlapping segments from the genome using the BED locations and the FASTA files:
```bash
cd ..
mkdir data/CSV
python 01_preprocessing/bed_to_csv.py --fasta_dir data/FASTA --bed_dir data/BED --output_dir data/CSV
```

After running bed_to_csv.py, your directory structure should look like this:
```
data/
├── BED/
├── CSV/
├── FASTA/
└── FragmentClassification/
```

## Fragment Classification

The directory `data/FragmentClassification` contains three files: `train.csv`, `test.csv`, `dev.csv`. Each file has two columns: `sequence`, `label`.

You can substitute these input files in any gLM binary classification pipeline.

Alternatively, you can download the dataset from HuggingFace:
[leannmlindsey/lambda](https://huggingface.co/datasets/leannmlindsey/lambda)

## Finetuning

See `02_finetuning/README.md` for instructions on setting up the finetuning environment, including cloning the Caduceus and DNABERT2 repositories.

## Inference Pipeline

### Running Individual Models

Each model has a wrapper script in `03_inference/` that can be submitted via SLURM:
```bash
sbatch 03_inference/wrapper_dnabert1_inference.sh
sbatch 03_inference/wrapper_dnabert2_inference.sh
sbatch 03_inference/wrapper_grover_inference.sh
sbatch 03_inference/wrapper_ntv2_inference.sh
sbatch 03_inference/wrapper_genalm_inference.sh
sbatch 03_inference/wrapper_evo_inference.sh
sbatch 03_inference/wrapper_prokbert_inference.sh
sbatch 03_inference/wrapper_megadna_inference.sh
```

### Running All Models with Orchestration Script

The orchestration script `run_all_inference.sh` submits all inference jobs for a dataset and organizes results in a timestamped directory:

```bash
# Run all 8 models on a dataset
./03_inference/run_all_inference.sh /path/to/dataset

# Run specific models only
./03_inference/run_all_inference.sh /path/to/dataset dnabert1 dnabert2 prokbert
```

**Features:**
- Creates timestamped results directory: `results/{dataset_name}_{YYYYMMDD_HHMMSS}/`
- Submits all jobs via sbatch
- Creates subdirectories for each model
- Saves logs to `{results_dir}/logs/`
- Creates `run_config.txt` documenting run parameters

**Available models:** `dnabert1`, `dnabert2`, `grover`, `ntv2`, `genalm`, `evo`, `prokbert`, `megadna`

### Results Structure

After running inference, results are organized as:
```
results/{dataset_name}_{timestamp}/
├── DNABERT1/
│   └── *.predictions.csv
├── DNABERT2/
│   └── *.predictions.csv
├── GROVER/
├── NTv2/
├── GenALM/
├── EVO/
├── ProkBERT/
├── megaDNA/
├── logs/
└── run_config.txt
```

## Instructions to Run Pretrained Bacterial Models (Caduceus)

### Setup Instructions

1. First, clone the Caduceus repository:
```bash
git clone https://github.com/kuleshov-group/caduceus.git
cd caduceus
```
Then follow instructions to set up the caduceus conda environment.

2. Create a directory for pretrained models (if it doesn't exist):
```bash
mkdir -p pretrained
cd pretrained
```

3. Extract the model files:
```bash
# Extract 4-layer Caduceus model
tar -xzf caduceus4L_B.tar.gz

# Extract 8-layer Caduceus model
tar -xzf caduceus8L_B.tar.gz

# Extract Hyena DNA model
tar -xzf Hyena_DNA_B.tar.gz
```

4. Copy the pretrained models to the appropriate directory:
```bash
# From your original pretrained_models directory
cp -r * /path/to/caduceus/pretrained/
```

After completing these steps, verify your directory structure looks like this:
```
caduceus/
├── pretrained/
│   ├── caduceus4L_B/
│   ├── caduceus8L_B/
│   └── Hyena_DNA_B/
└── ... (other caduceus files)
```

You can then use these models in the same way described in the Caduceus GitHub, just provide the correct paths to `config.json` and `weights.ckpt`.

## Genome Wide Prophage Detection - Raw Score

You will find example scripts for processing the test genome CSV files in the `03_inference/` folder.

## Genome Wide Prophage Detection - Clustered Score

After inference you will need to run the signal extraction.

1. First compile the C++ scripts:
```bash
cd 04_signal_extraction/src
g++ -std=c++17 prophage_signal_processor.cpp -o prophage_signal_processor -lstdc++fs -pthread
```

2. Run signal extraction on predictions:
```bash
cd 04_signal_extraction
./run_psp_all.sh /path/to/predictions
```

This applies multiple signal extraction algorithms (moving window average, run length encoding, DBSCAN, median filter, connected component labeling) and calculates performance metrics.

## Citation

If you use Lambda in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]
