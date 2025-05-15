#!/bin/bash

module load seqkit/2.8.2

# Check for correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <parent_directory> <output_file>"
    exit 1
fi

# Input directory and output file
parent_dir="$1"
output_file="$2"

# Check if the output file already exists, and delete if so
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

for sub_dir in "$parent_dir"/*; do
    echo "*******************************************"
    echo "sub_dir:" $sub_dir
    echo "*******************************************"
    
    # Check if it is a directory
    if [ -d "$sub_dir" ]; then
        # Find the .fna file within the subdirectory
        #fasta_file=$(find "$sub_dir" -type f -name "*.fna" | head -n 1)
        fasta_file=$(find "$sub_dir" -type f \( -name "*.fna" -o -name "*.fasta" \) | head -n 1)
	echo "fasta_file:" $fasta_file

        # Check if the .fna file exists and is not empty
        if [ -n "$fasta_file" ] && [ -s "$fasta_file" ]; then
            # Calculate GC content using seqkit
            gc_content=$(seqkit fx2tab -n -g "$fasta_file" | sed 's/ /	/')  # Replace the first space with a tab
            
            # Loop through each line of gc_content and prepend basename
            while IFS= read -r line; do
                echo -e "$(basename "$sub_dir")\t$line"
                echo -e "$(basename "$sub_dir")\t$line" >> "$output_file"
            done <<< "$gc_content"
        
        else
            echo "$(basename "$sub_dir"): No .fna file found or file is empty"
        fi
    fi
done

echo "GC content calculation complete. Results saved to $output_file."
