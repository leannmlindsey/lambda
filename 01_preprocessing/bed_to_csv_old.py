import csv
from Bio import SeqIO
import os
from pathlib import Path
import argparse

def parse_bed(bed_file):
    regions = []
    with open(bed_file, 'r') as bed:
        for line in bed:
            if line.startswith('#') or not line.strip():
                continue
            chrom, start, end = line.strip().split()[:3]
            regions.append((chrom, int(start), int(end)))
    return regions

def extract_sequences_to_csv(fasta_file, bed_file, output_file):
    sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    regions = parse_bed(bed_file)

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['SeqID', 'start', 'end', 'sequence', 'label'])

        for chrom, start, end in regions:
            if chrom in sequences:
                seq = sequences[chrom].seq[start:end]
                # Write the row with SeqID, start, end, sequence, and label 0
                csvwriter.writerow([chrom, start, end, str(seq), 0])
            else:
                print(f"Warning: Chromosome {chrom} not found in FASTA file.")

def process_directories(fasta_dir, bed_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all FASTA files
    fasta_files = Path(fasta_dir).glob('*.fna')
    
    for fasta_file in fasta_files:
        # Get the base name without extension
        base_name = fasta_file.stem
        
        # Construct corresponding BED and output CSV file paths
        bed_file = Path(bed_dir) / f"{base_name}.bed"
        output_file = Path(output_dir) / f"{base_name}.csv"
        
        # Check if corresponding BED file exists
        if bed_file.exists():
            print(f"Processing {base_name}...")
            try:
                extract_sequences_to_csv(str(fasta_file), str(bed_file), str(output_file))
                print(f"Successfully processed {base_name}")
            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
        else:
            print(f"Warning: No matching BED file found for {base_name}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Extract sequences from FASTA files using BED coordinates and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
    python bed_to_csv.py --fasta_dir /path/to/fasta/files --bed_dir /path/to/bed/files --output_dir /path/to/output

This script will:
1. Find all .fna files in the fasta_dir
2. Look for corresponding .bed files with matching names in bed_dir
3. Extract sequences and save them as .csv files in output_dir
        '''
    )
    
    parser.add_argument(
        '--fasta_dir',
        required=True,
        help='Directory containing FASTA files (*.fna)'
    )
    
    parser.add_argument(
        '--bed_dir',
        required=True,
        help='Directory containing BED files (*.bed)'
    )
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where CSV files will be saved'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    process_directories(args.fasta_dir, args.bed_dir, args.output_dir)
