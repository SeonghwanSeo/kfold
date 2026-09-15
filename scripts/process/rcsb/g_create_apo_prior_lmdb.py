"""Build protein and protein-multimer apo/prior LMDBs from AtlasFold outputs."""

from kfold.training.preprocess.apo_lmdb import main

if __name__ == "__main__":
    main()
