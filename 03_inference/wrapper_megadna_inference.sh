#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=inference_megadna
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH -o ../watch/%x%j.outerror

mkdir -p ../watch

# Load Modules
nvidia-smi

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMBDA_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${LAMBDA_DIR}/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# =============================================================================
# SETUP ENVIRONMENT
# =============================================================================

CONDA_ENV=${CONDA_ENV_MEGADNA:-dna}
echo "Activating conda environment: $CONDA_ENV"
source activate $CONDA_ENV
conda list

# =============================================================================
# SET PATHS
# =============================================================================

input_dir=${INPUT_DIR:-${DEFAULT_INPUT_DIR:-$LAMBDA_DIR/data/CSV}}
output_dir=${OUTPUT_DIR:-$LAMBDA_DIR/results/megaDNA}
script_dir=$LAMBDA_DIR/03_inference
model_ckpt_dir=${MODEL_CKPT_DIR:-${CKPT_MEGADNA:-$LAMBDA_DIR/phage_finetuned_model_ckpts/megaDNA}}

mkdir -p $output_dir

echo "=============================================="
echo "megaDNA Inference"
echo "=============================================="
echo "Input Dir:       $input_dir"
echo "Output Dir:      $output_dir"
echo "Model Checkpoint: $model_ckpt_dir"
echo "=============================================="

cd $input_dir

for input_file in *; do
    basename="${input_file%.*}"
    echo "Processing: $input_file"
    echo "Output File: $output_dir/$basename.predictions.csv"
    python $script_dir/megadna_inference.py $input_dir/$input_file $output_dir/$basename.predictions.csv $model_ckpt_dir
done

echo "megaDNA inference complete."
