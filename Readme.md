# Lambda Benchmark Installation Guide

## Overview
Lambda (LAnguage Model Bacteriophage Detection Assessment) is a benchmark for testing genomic language models in bacterial domain tasks. This guide will help you set up the required datasets.

### Note: The datasets and pretrained models are available on huggingface but we were unable to include them in this repository because of space limitations. We also hosted them at Zenodo, but they are still under review so we are unable to send a link at this time. We also attempted to load them onto Github Anonymous, but they were too large. We are sorry for the inconvenience.


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

```bash
tar -xf BED.tar
tar -xzf BinaryClassification.tar.gz
tar -xf FASTA.tar
```

After extraction, your directory structure should look like this:
```
data/
├── BED/
├── BinaryClassification/
└── FASTA/

4. Prepare the genomes for processing

```bash
cd ..
mkdir CSV
python preprocessing/bed_to_csv.py --fasta_dir data/FASTA --fasta_dir BED --output_dir CSV 
```
# Task 1: Binary Classification 

The directory data/BinaryClassification has three files in it train.csv, test.csv, dev.csv.  Each file has two columns, sequence,label.  

You can substitute these input files in any gLM binary classification pipeline.

# Pretrained Bacterial Models

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

After completing these steps, verify your directory structure looks like this:
```
caduceus/
├── pretrained/
│   ├── caduceus4L_B/
│   ├── caduceus8L_B/
│   └── Hyena_DNA_B/
└── ... (other caduceus files)
```

You can then use these models in the same way described in the Caduceus github, just provide the correct paths to config.json and weights.ckpt.

## Task 2: Prophage Detection 

You will find example scripts for processing the test genome CSV files in the inference folder

## Signal Extraction

After inference you will need to run the signal extraction 

1. First compile the C++ scripts
```bash
g++ -std=c++17 process_csv.cpp -o process_csv -lstdc++fs
g++ -std=c++17 prophage_signal_processor.cpp -o prophage_signal_processor -lstdc++fs -pthread
```




