#!/bin/bash
#
# Lambda Configuration File
#
# This file contains all configurable paths for the Lambda benchmark pipeline.
# Edit these paths to match your local setup.
#
# Usage: Source this file in your scripts
#   source /path/to/lambda/config.sh
#

# =============================================================================
# BASE DIRECTORIES
# =============================================================================

# Root directory of the Lambda repository
# Update this to your actual Lambda installation path
LAMBDA_DIR="/data/lindseylm/gLMs/lambda"

# Default input data directory (CSV files with sequences)
DEFAULT_INPUT_DIR="${LAMBDA_DIR}/data/CSV"

# Default results directory (predictions will be saved here)
DEFAULT_RESULTS_DIR="${LAMBDA_DIR}/results"

# =============================================================================
# CONDA ENVIRONMENTS
# =============================================================================
# Specify the conda environment name for each model
# These can all be the same if you have a unified environment

CONDA_ENV_DNABERT1="dna"
CONDA_ENV_DNABERT2="dna"
CONDA_ENV_GROVER="dna"
CONDA_ENV_NTV2="dna"
CONDA_ENV_GENALM="dna"
CONDA_ENV_EVO="dna"
CONDA_ENV_PROKBERT="dna"
CONDA_ENV_MEGADNA="dna"
CONDA_ENV_CADUCEUS="caduceus"
CONDA_ENV_HYENADNA="caduceus"

# =============================================================================
# MODEL CHECKPOINT DIRECTORIES
# =============================================================================
# Specify the path to finetuned model checkpoints for each model
# These can point to different locations depending on your setup

# Option 1: All checkpoints in a central directory
CKPT_BASE_DIR="${LAMBDA_DIR}/phage_finetuned_model_ckpts"

# Option 2: Checkpoints in the finetuning directory (e.g., after cloning repos)
# CKPT_BASE_DIR="${LAMBDA_DIR}/02_finetuning"

# Individual model checkpoint paths
# Modify these if your checkpoints are in non-standard locations
CKPT_DNABERT1="${CKPT_BASE_DIR}/DNABERT1"
CKPT_DNABERT2="${CKPT_BASE_DIR}/DNABERT2"
CKPT_GROVER="${CKPT_BASE_DIR}/GROVER"
CKPT_NTV2="${CKPT_BASE_DIR}/NTv2"
CKPT_GENALM="${CKPT_BASE_DIR}/GenALM"
CKPT_EVO="${CKPT_BASE_DIR}/EVO"
CKPT_PROKBERT="${CKPT_BASE_DIR}/ProkBERT"
CKPT_MEGADNA="${CKPT_BASE_DIR}/megaDNA"

# Caduceus models (if using caduceus from 02_finetuning)
CKPT_CADUCEUS_4L="${LAMBDA_DIR}/02_finetuning/caduceus/pretrained/caduceus4L_B"
CKPT_CADUCEUS_8L="${LAMBDA_DIR}/02_finetuning/caduceus/pretrained/caduceus8L_B"
CKPT_HYENADNA="${LAMBDA_DIR}/02_finetuning/caduceus/pretrained/Hyena_DNA_B"

# =============================================================================
# INFERENCE SCRIPT DIRECTORIES
# =============================================================================

SCRIPT_DIR="${LAMBDA_DIR}/03_inference"

# =============================================================================
# SLURM SETTINGS (for HPC clusters)
# =============================================================================

SLURM_PARTITION="soc-gpu-np"
SLURM_ACCOUNT="soc-gpu-np"
SLURM_TIME="4:00:00"
SLURM_MEM="64G"
SLURM_GPU="gpu:a6000:1"

# =============================================================================
# HELPER FUNCTION: Get checkpoint path for a model
# =============================================================================

get_checkpoint_path() {
    local model=$1
    case "$model" in
        dnabert1)   echo "$CKPT_DNABERT1" ;;
        dnabert2)   echo "$CKPT_DNABERT2" ;;
        grover)     echo "$CKPT_GROVER" ;;
        ntv2)       echo "$CKPT_NTV2" ;;
        genalm)     echo "$CKPT_GENALM" ;;
        evo)        echo "$CKPT_EVO" ;;
        prokbert)   echo "$CKPT_PROKBERT" ;;
        megadna)    echo "$CKPT_MEGADNA" ;;
        caduceus4l) echo "$CKPT_CADUCEUS_4L" ;;
        caduceus8l) echo "$CKPT_CADUCEUS_8L" ;;
        hyenadna)   echo "$CKPT_HYENADNA" ;;
        *)          echo "" ;;
    esac
}

# =============================================================================
# HELPER FUNCTION: Get conda environment for a model
# =============================================================================

get_conda_env() {
    local model=$1
    case "$model" in
        dnabert1)   echo "$CONDA_ENV_DNABERT1" ;;
        dnabert2)   echo "$CONDA_ENV_DNABERT2" ;;
        grover)     echo "$CONDA_ENV_GROVER" ;;
        ntv2)       echo "$CONDA_ENV_NTV2" ;;
        genalm)     echo "$CONDA_ENV_GENALM" ;;
        evo)        echo "$CONDA_ENV_EVO" ;;
        prokbert)   echo "$CONDA_ENV_PROKBERT" ;;
        megadna)    echo "$CONDA_ENV_MEGADNA" ;;
        caduceus4l) echo "$CONDA_ENV_CADUCEUS" ;;
        caduceus8l) echo "$CONDA_ENV_CADUCEUS" ;;
        hyenadna)   echo "$CONDA_ENV_HYENADNA" ;;
        *)          echo "dna" ;;
    esac
}

# =============================================================================
# HELPER FUNCTION: Get model display name
# =============================================================================

get_model_dir_name() {
    local model=$1
    case "$model" in
        dnabert1)   echo "DNABERT1" ;;
        dnabert2)   echo "DNABERT2" ;;
        grover)     echo "GROVER" ;;
        ntv2)       echo "NTv2" ;;
        genalm)     echo "GenALM" ;;
        evo)        echo "EVO" ;;
        prokbert)   echo "ProkBERT" ;;
        megadna)    echo "megaDNA" ;;
        caduceus4l) echo "Caduceus4L" ;;
        caduceus8l) echo "Caduceus8L" ;;
        hyenadna)   echo "HyenaDNA" ;;
        *)          echo "$model" ;;
    esac
}
