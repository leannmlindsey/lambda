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
#   ./run_all_inference.sh /path/to/dataset --list             # List available models
#

set -e

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMBDA_DIR="$(dirname "$SCRIPT_DIR")"

# Source the configuration file
CONFIG_FILE="${LAMBDA_DIR}/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo "Loaded configuration from: $CONFIG_FILE"
else
    echo "Warning: Config file not found at $CONFIG_FILE"
    echo "Using default paths. Create config.sh to customize paths."
    # Set defaults if config not found
    LAMBDA_DIR="${LAMBDA_DIR}"
    SCRIPT_DIR="${LAMBDA_DIR}/03_inference"
fi

# =============================================================================
# DEFINE AVAILABLE MODELS
# =============================================================================

# All available models (add new models here)
ALL_MODELS="dnabert1 dnabert2 grover ntv2 genalm evo prokbert megadna"

# Models that have working inference scripts (update as you add scripts)
WORKING_MODELS="dnabert1 dnabert2 grover ntv2 genalm"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

show_help() {
    echo "Lambda Inference Pipeline - Orchestration Script"
    echo ""
    echo "Usage: $0 <dataset_path> [options] [models...]"
    echo ""
    echo "Arguments:"
    echo "  dataset_path    Path to directory containing input CSV files"
    echo "  models          Space-separated list of models to run (optional)"
    echo ""
    echo "Options:"
    echo "  --list          List all available models and their status"
    echo "  --working       Run only models with working inference scripts"
    echo "  --all           Run all models (default if no models specified)"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Available models: $ALL_MODELS"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/dataset                      # Run all models"
    echo "  $0 /path/to/dataset dnabert1 dnabert2    # Run specific models"
    echo "  $0 /path/to/dataset --working            # Run only working models"
    echo "  $0 --list                                # List models and status"
}

list_models() {
    echo "=============================================="
    echo "Available Models and Status"
    echo "=============================================="
    echo ""
    printf "%-12s %-20s %-15s %s\n" "MODEL" "CHECKPOINT PATH" "WRAPPER" "INFERENCE SCRIPT"
    echo "----------------------------------------------------------------------"

    for model in $ALL_MODELS; do
        # Get checkpoint path
        if type get_checkpoint_path &>/dev/null; then
            ckpt_path=$(get_checkpoint_path "$model")
        else
            ckpt_path="(config not loaded)"
        fi

        # Check if checkpoint exists
        if [ -d "$ckpt_path" ]; then
            ckpt_status="EXISTS"
        else
            ckpt_status="MISSING"
        fi

        # Check wrapper script
        wrapper="${SCRIPT_DIR}/wrapper_${model}_inference.sh"
        if [ -f "$wrapper" ]; then
            wrapper_status="OK"
        else
            wrapper_status="MISSING"
        fi

        # Check inference script
        inference="${SCRIPT_DIR}/${model}_inference.py"
        if [ -f "$inference" ]; then
            inference_status="OK"
        else
            # Check for .sh variant
            inference="${SCRIPT_DIR}/${model}_inference.sh"
            if [ -f "$inference" ]; then
                inference_status="OK (.sh)"
            else
                inference_status="MISSING"
            fi
        fi

        printf "%-12s %-20s %-15s %s\n" "$model" "$ckpt_status" "$wrapper_status" "$inference_status"
    done

    echo ""
    echo "Checkpoint base directory: ${CKPT_BASE_DIR:-not set}"
    echo "Script directory: $SCRIPT_DIR"
    echo ""
    echo "To customize paths, edit: $CONFIG_FILE"
}

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

# Handle special flags first
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --list)
        list_models
        exit 0
        ;;
esac

# Check minimum arguments
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

DATASET_PATH="$1"
shift

# Handle options and collect models
MODELS=""
USE_WORKING_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --working)
            USE_WORKING_ONLY=true
            shift
            ;;
        --all)
            MODELS="$ALL_MODELS"
            shift
            ;;
        --*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            # It's a model name
            MODELS="$MODELS $1"
            shift
            ;;
    esac
done

# Set default models if none specified
if [ -z "$MODELS" ]; then
    if [ "$USE_WORKING_ONLY" = true ]; then
        MODELS="$WORKING_MODELS"
    else
        MODELS="$ALL_MODELS"
    fi
fi

# Trim leading/trailing whitespace
MODELS=$(echo "$MODELS" | xargs)

# =============================================================================
# VALIDATE INPUTS
# =============================================================================

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

# =============================================================================
# SETUP OUTPUT DIRECTORIES
# =============================================================================

# Create parent results directory with dataset name and timestamp
RESULTS_PARENT="${DEFAULT_RESULTS_DIR:-${LAMBDA_DIR}/results}/${DATASET_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_PARENT"
mkdir -p "${RESULTS_PARENT}/logs"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Lambda Inference Pipeline"
echo "=============================================="
echo "Dataset:         $DATASET_NAME"
echo "Dataset Path:    $DATASET_PATH"
echo "Results Dir:     $RESULTS_PARENT"
echo "Timestamp:       $TIMESTAMP"
echo "Config File:     $CONFIG_FILE"
echo "=============================================="
echo "Models to run:   $MODELS"
echo "=============================================="

# =============================================================================
# SUBMIT JOBS
# =============================================================================

SUBMITTED_COUNT=0
SKIPPED_COUNT=0

for MODEL in $MODELS; do
    # Get model directory name using helper function or fallback
    if type get_model_dir_name &>/dev/null; then
        MODEL_DIR=$(get_model_dir_name "$MODEL")
    else
        # Fallback if config not loaded
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
                ((SKIPPED_COUNT++))
                continue
                ;;
        esac
    fi

    # Get checkpoint path using helper function or fallback
    if type get_checkpoint_path &>/dev/null; then
        MODEL_CKPT=$(get_checkpoint_path "$MODEL")
    else
        MODEL_CKPT="${LAMBDA_DIR}/phage_finetuned_model_ckpts/${MODEL_DIR}"
    fi

    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_${MODEL}_inference.sh"

    # Check if wrapper script exists
    if [ ! -f "$WRAPPER_SCRIPT" ]; then
        echo "Warning: Wrapper script not found for $MODEL: $WRAPPER_SCRIPT"
        ((SKIPPED_COUNT++))
        continue
    fi

    # Create model-specific output directory
    MODEL_OUTPUT_DIR="${RESULTS_PARENT}/${MODEL_DIR}"
    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "Submitting $MODEL..."
    echo "  Checkpoint:  $MODEL_CKPT"
    echo "  Output Dir:  $MODEL_OUTPUT_DIR"

    # Check if checkpoint exists (warning only, still submit)
    if [ ! -d "$MODEL_CKPT" ]; then
        echo "  Warning: Checkpoint directory not found (job may fail)"
    fi

    # Submit job with environment variables
    JOB_ID=$(sbatch \
        --export=ALL,INPUT_DIR="$DATASET_PATH",OUTPUT_DIR="$MODEL_OUTPUT_DIR",MODEL_CKPT_DIR="$MODEL_CKPT" \
        --job-name="inf_${MODEL}_${DATASET_NAME}" \
        --output="${RESULTS_PARENT}/logs/${MODEL}_%j.out" \
        --error="${RESULTS_PARENT}/logs/${MODEL}_%j.err" \
        "$WRAPPER_SCRIPT" 2>&1 | awk '{print $4}')

    if [ -n "$JOB_ID" ]; then
        echo "  Job ID: $JOB_ID"
        ((SUBMITTED_COUNT++))
    else
        echo "  Error: Failed to submit job"
        ((SKIPPED_COUNT++))
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================

echo "=============================================="
echo "Submission Complete"
echo "=============================================="
echo "Jobs submitted: $SUBMITTED_COUNT"
echo "Jobs skipped:   $SKIPPED_COUNT"
echo ""
echo "Results will be written to: $RESULTS_PARENT"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "=============================================="

# Save run configuration
cat > "${RESULTS_PARENT}/run_config.txt" << EOF
Lambda Inference Run Configuration
==================================
Date: $(date)
Dataset Name: $DATASET_NAME
Dataset Path: $DATASET_PATH
Timestamp: $TIMESTAMP
Models Requested: $MODELS
Jobs Submitted: $SUBMITTED_COUNT
Jobs Skipped: $SKIPPED_COUNT
Results Directory: $RESULTS_PARENT
Config File: $CONFIG_FILE

Model Checkpoint Paths:
EOF

for MODEL in $MODELS; do
    if type get_checkpoint_path &>/dev/null; then
        MODEL_CKPT=$(get_checkpoint_path "$MODEL")
    else
        MODEL_CKPT="(config not loaded)"
    fi
    echo "  $MODEL: $MODEL_CKPT" >> "${RESULTS_PARENT}/run_config.txt"
done

echo ""
echo "Run configuration saved to: ${RESULTS_PARENT}/run_config.txt"
