# RFAM Preprocessing

RFAM RNA monomer distillation uses masked trunk apo inputs and a
Langevin-generated prior.

## Sequence Extraction

```bash
.venv/bin/python scripts/process/rfam/a1_extract_sequences.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3
```

This writes:

```text
rfam/sequences/rfam_sequences.fasta
```
