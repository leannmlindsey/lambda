# Lambda Benchmark Installation Guide

## Overview
Lambda (LAnguage Model Bacteriophage Detection Assessment) is a benchmark for testing genomic language models in bacterial domain tasks. This guide will help you set up the required datasets.

## Prerequisites
- Unix-like operating system (Linux/macOS)
- Command line access
- Basic understanding of terminal commands
- Sufficient disk space for the extracted files

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/[repository-name]/lambda.git
cd lambda
```

2. Navigate to the data directory:
```bash
cd data
```

3. Extract the archive files:

For BED.tar:
```bash
tar -xf BED.tar
```

For BinaryClassification.tar.gz:
```bash
tar -xzf BinaryClassification.tar.gz
```

For FASTA.tar:
```bash
tar -xf FASTA.tar
```

## Directory Structure
After extraction, your directory structure should look like this:
```
data/
├── BED/
├── BinaryClassification/
└── FASTA/
```
# Prepare Pretrained Bacterial Models

## Directory Structure
Before starting, ensure you have the following files:
```
pretrained_models/
├── caduceus4L_B.tar.gz
├── caduceus8L_B.tar.gz
└── Hyena_DNA_B.tar.gz
```

## Setup Instructions

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

## Verification
After completing these steps, verify your directory structure looks like this:
```
caduceus/
├── pretrained/
│   ├── caduceus4L_B/
│   ├── caduceus8L_B/
│   └── Hyena_DNA_B/
└── ... (other caduceus files)
```

