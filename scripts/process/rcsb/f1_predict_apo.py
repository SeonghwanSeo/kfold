"""Predict seed-specific AtlasFold monomer apo ensembles (configurable maximum length)."""

from kfold.training.preprocess.atlasfold_prediction import main

if __name__ == "__main__":
    main("protein")
