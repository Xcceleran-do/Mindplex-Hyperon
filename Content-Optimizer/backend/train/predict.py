"""Predict helper: loads a model and scores candidate feature vectors (placeholder)."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models')
    args = parser.parse_args()
    print('This is a placeholder for predict.py. Model dir:', args.model)
