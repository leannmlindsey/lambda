#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=inference_genalm
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH -o ../watch/%x%j.outerror

mkdir ../watch

# Load Modules 
nvidia-smi

echo "starting DNABERT2 env on conda"
source activate dna
conda list

lambda_dir="/data/lindseylm/gLMs/lambda"
input_dir=$lambda_dir/data/CSV      # Place the full path to the CSV directory here
output_dir=$lambda_dir/results/GENALM     # Place the full path to the results directory here
script_dir=$lambda_dir/03_inference     # Place the full path to the inference directory here
model_ckpt_dir=$lambda_dir/phage_finetuned_model_ckpts/GENALM
mkdir $output_dir

cd $input_dir

for input_file in *; do
    basename="${input_file%.*}"
    echo "File is $input_dir/$input_file"
    echo "Basename is $basename"
    echo "Output File is $output_dir/$basename.predictions.csv"
    python $script_dir/genalm_inference.py $input_dir/$input_file $output_dir/$basename.predictions.csv $model_ckpt_dir
done

