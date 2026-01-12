#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=inference_evo
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH -o ../watch/%x%j.outerror

mkdir -p ../watch

# Load Modules
nvidia-smi

echo "starting conda env"
source activate dna
conda list

# Use environment variables if set, otherwise use defaults
lambda_dir="/data/lindseylm/gLMs/lambda"
input_dir=${INPUT_DIR:-$lambda_dir/data/CSV}
output_dir=${OUTPUT_DIR:-$lambda_dir/results/EVO}
script_dir=$lambda_dir/03_inference
model_ckpt_dir=${MODEL_CKPT_DIR:-$lambda_dir/phage_finetuned_model_ckpts/EVO}

mkdir -p $output_dir

echo "Input Dir: $input_dir"
echo "Output Dir: $output_dir"
echo "Model Checkpoint: $model_ckpt_dir"

cd $input_dir

for input_file in *; do
    basename="${input_file%.*}"
    echo "File is $input_dir/$input_file"
    echo "Basename is $basename"
    echo "Output File is $output_dir/$basename.predictions.csv"
    python $script_dir/evo_inference.py $input_dir/$input_file $output_dir/$basename.predictions.csv $model_ckpt_dir
done

