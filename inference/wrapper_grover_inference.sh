#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=inference_grover
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH -o ../watch/%x%j.outerror

mkdir ../watch
# Load Modules 
nvidia-smi
module load cuda
nvidia-smi

echo "starting DNABERT2 env on conda"
source activate dna_sandbox    # replace this with your DNABERT2 conda environment name
conda list


input_dir=      # Place the full path to the CSV directory here
output_dir=     # Place the full path to the results directory here
script_dir=     # Place the full path to the inference directory here
mkdir $output_dir

cd $input_dir

for input_file in *; do
    basename="${input_file%.*}"
    echo "File is $input_dir/$input_file"
    echo "Basename is $basename"
    echo "Output File is $output_dir/$basename.predictions.csv"
    python $script_dir/grover_inference.py $input_dir/$input_file $output_dir/$basename.predictions.csv
done

