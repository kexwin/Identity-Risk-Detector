import os
import argparse
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Identity Risk Detector Pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory containing input CSVs')
    parser.add_argument('--output-dir', default='output', help='Directory for output reports')
    args = parser.parse_args()
    
    run_pipeline(args.data_dir, args.output_dir)
