"""Predict AtlasFold-m apo ensembles from explicit chain-group CSV metadata."""

from kfold.training.preprocess.atlasfold_prediction import main

if __name__ == "__main__":
    main("protein-multimer")
