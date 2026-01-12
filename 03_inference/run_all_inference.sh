#!/bin/bash
#
# Orchestration script to submit all inference jobs for a given dataset
#
# Usage: ./run_all_inference.sh <dataset_path> [models...]
#
# Arguments:
#   dataset_path  - Path to the directory containing input CSV files
#   models        - Optional: space-separated list of models to run
#                   If not specified, runs all models
#
# Available models: dnabert1, dnabert2, grover, ntv2, genalm, evo, prokbert, megadna
#
# Examples:
#   ./run_all_inference.sh /path/to/dataset                    # Run all models
#   ./run_all_inference.sh /path/to/dataset dnabert1 dnabert2  # Run specific models
#

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [models...]"
    echo ""
    echo "Available models: dnabert1, dnabert2, grover, ntv2, genalm, evo, prokbert, megadna"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/dataset                    # Run all models"
    echo "  $0 /path/to/dataset dnabert1 dnabert2  # Run specific models"
    exit 1
fi

DATASET_PATH="$1"
shift

# Validate dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Get absolute path
DATASET_PATH=$(cd "$DATASET_PATH" && pwd)

# Extract dataset name from path
DATASET_NAME=$(basename "$DATASET_PATH")

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define base directories
LAMBDA_DIR="/data/lindseylm/gLMs/lambda"
SCRIPT_DIR="$LAMBDA_DIR/03_inference"
MODEL_CKPT_BASE="$LAMBDA_DIR/phage_finetuned_model_ckpts"

# Create parent results directory with dataset name and timestamp
RESULTS_PARENT="${LAMBDA_DIR}/results/${DATASET_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_PARENT"

echo "=============================================="
echo "Lambda Inference Pipeline"
echo "=============================================="
echo "Dataset:         $DATASET_NAME"
echo "Dataset Path:    $DATASET_PATH"
echo "Results Dir:     $RESULTS_PARENT"
echo "Timestamp:       $TIMESTAMP"
echo "=============================================="

# Define all available models
ALL_MODELS="dnabert1 dnabert2 grover ntv2 genalm evo prokbert megadna"

# Use specified models or all models
if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="$ALL_MODELS"
fi

echo "Models to run: $MODELS"
echo "=============================================="

# Submit jobs for each model
for MODEL in $MODELS; do
    # Convert model name to uppercase for directory names
    MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')

    # Handle special cases for directory naming
    case "$MODEL" in
        dnabert1) MODEL_DIR="DNABERT1" ;;
        dnabert2) MODEL_DIR="DNABERT2" ;;
        grover)   MODEL_DIR="GROVER" ;;
        ntv2)     MODEL_DIR="NTv2" ;;
        genalm)   MODEL_DIR="GenALM" ;;
        evo)      MODEL_DIR="EVO" ;;
        prokbert) MODEL_DIR="ProkBERT" ;;
        megadna)  MODEL_DIR="megaDNA" ;;
        *)
            echo "Warning: Unknown model '$MODEL', skipping..."
            continue
            ;;
    esac

    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_${MODEL}_inference.sh"

    # Check if wrapper script exists
    if [ ! -f "$WRAPPER_SCRIPT" ]; then
        echo "Warning: Wrapper script not found for $MODEL: $WRAPPER_SCRIPT"
        continue
    fi

    # Create model-specific output directory
    MODEL_OUTPUT_DIR="${RESULTS_PARENT}/${MODEL_DIR}"
    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "Submitting $MODEL..."
    echo "  Output Dir: $MODEL_OUTPUT_DIR"

    # Submit job with environment variables
    JOB_ID=$(sbatch \
        --export=ALL,INPUT_DIR="$DATASET_PATH",OUTPUT_DIR="$MODEL_OUTPUT_DIR",MODEL_CKPT_DIR="${MODEL_CKPT_BASE}/${MODEL_DIR}" \
        --job-name="inf_${MODEL}_${DATASET_NAME}" \
        --output="${RESULTS_PARENT}/logs/${MODEL}_%j.out" \
        --error="${RESULTS_PARENT}/logs/${MODEL}_%j.err" \
        "$WRAPPER_SCRIPT" | awk '{print $4}')

    echo "  Job ID: $JOB_ID"
done

# Create logs directory
mkdir -p "${RESULTS_PARENT}/logs"

echo "=============================================="
echo "All jobs submitted!"
echo "Results will be written to: $RESULTS_PARENT"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "=============================================="

# Save run configuration
cat > "${RESULTS_PARENT}/run_config.txt" << EOF
Lambda Inference Run Configuration
==================================
Dataset Name: $DATASET_NAME
Dataset Path: $DATASET_PATH
Timestamp: $TIMESTAMP
Models: $MODELS
Results Directory: $RESULTS_PARENT
EOF

echo "Run configuration saved to: ${RESULTS_PARENT}/run_config.txt"
