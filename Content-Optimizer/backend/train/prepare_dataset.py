"""Prepare dataset: placeholder script that will fetch embeddings from Neo4j and assemble training data."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/dataset.parquet')
    args = parser.parse_args()
    print('This is a placeholder for prepare_dataset.py. Output would be:', args.out)
