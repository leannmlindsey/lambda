# Lambda Benchmark Installation Guide

## Overview
Lambda (LAnguage Model Bacteriophage Detection Assessment) is a benchmark for testing genomic language models in bacterial domain tasks. This guide will help you set up the required datasets.

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/leannmlindsey/lambda.git
cd lambda
```

2. Download the datasets and upload them and place them into the data directory

3. Navigate to the data directory:
```bash
cd data
```

4. Extract the archive files:

```bash
tar -xf BED.tar
tar -xf FASTA.tar
tar -xf FragmentClassification
```

After extraction, your directory structure should look like this:
```
data/
├── BED/
└── FASTA/
|-- FragmentClassification/

5. Prepare the genomes for processing

```bash
cd ..
mkdir CSV
python preprocessing/bed_to_csv.py --fasta_dir data/FASTA --bed_dir BED --output_dir data/CSV 
```
# Fragment Classification 

The directory data/FragmentClassification has three files in it train.csv, test.csv, dev.csv.  Each file has two columns, sequence,label.  

You can substitute these input files in any gLM binary classification pipeline.

Alternately, you can download the dataset from huggingface:

[leannmlindsey/lambda](https://huggingface.co/datasets/leannmlindsey/lambda)

# Instructions to run the Pretrained Bacterial Models

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

## Genome Wide Prophage Detection - Raw Score 

You will find example scripts for processing the test genome CSV files in the inference folder

## Genome Wide Prophage Detection - Clustered Score

After inference you will need to run the signal extraction 

1. First compile the C++ scripts
```bash
g++ -std=c++17 process_csv.cpp -o process_csv -lstdc++fs
g++ -std=c++17 prophage_signal_processor.cpp -o prophage_signal_processor -lstdc++fs -pthread
```




