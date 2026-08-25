# K-Fold

K-Fold is a biomolecular foundation model for high-accuracy structure prediction of biomolecular complexes.
K-Fold leverages pre-trained sequence encoders and structure encoders to achieve high fidelity **without Multiple Sequence Alignments (MSAs)**.
By incorporating an **apo-to-holo diffusion bridge**, K-Fold models the transition from unbound to bound states, serving as a unified and extensible research tool for the computational biology community.

---

## Installation

K-Fold requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/wykim-lab/kfold.git
cd kfold

# Install with cuequivariance kernels
pip install '.[cuequiv]'

# Install in editable mode with training/dev dependencies
pip install -e '.[train,cuequiv,dev]'
pre-commit install
```

## Inference

K-Fold takes a YAML or JSON input file defining the molecular entities and their sequences/apo structures.

```bash
# Single-GPU Inference
python scripts/inference.py \
  --config configs/kfold.yaml \
  --checkpoint path/to/model.ckpt \
  --input examples/casp15_h1106.yaml \
  --out_dir ./results/
```

For detailed instructions on input formats (SMILES, CCD, apo paths) and multi-GPU execution, see the **[Inference Guide](docs/inference.md)**.

> Future Work
> - **Automatic Apo Prediction**: Automatically call ESMFold to generate protein apo structures if not provided in the query.
> - **Improved Folding Model**: Develop and integrate a superior protein folding model to provide high-quality apo structures automatically.

---

## Training & Development

K-Fold is designed to be highly extensible for structural biology research.

- **Data Structures**: Model input and structure representations are documented in **[Data Structure](docs/developers/DATA_STRUCTURE.md)**.
- **Training**: Detailed instructions for dataset preparation and training loops are available in the **[Training Guide](docs/developers/TRAINING_GUIDE.md)**.

---

## License

This project is licensed under the terms of the Apache 2.0 license. See `LICENSE` for more details.

## Citation

If you use K-Fold in your research, please cite:

```text
(TODO: Add citation here)
```
