"""Train ranker: placeholder script that trains a LightGBM regressor on prepared features."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/dataset.parquet')
    parser.add_argument('--out_dir', default='./models')
    args = parser.parse_args()
    print('This is a placeholder for train_ranker.py. Input:', args.input, 'Out dir:', args.out_dir)
