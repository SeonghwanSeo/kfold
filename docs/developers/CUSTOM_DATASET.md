# Custom distillation datasets

K-Fold's training datasets are not distributed with the release. Prepare your
own predicted structures using the common `DistillationDataset`, selected with
`type: distillation`. Protein and nucleic-acid monomers, dimers, and complexes
containing ligands all use the same loader. Dataset names are user-defined.
Experimental RCSB structures use `type: rcsb` for experimental confidence-head
supervision.

## Storage contract

Each dataset directory contains:

```text
my_predictions/
  structure.lmdb/
  manifest.msgpack
```

- Each LMDB key is the UTF-8 encoding of `RefStructure.metadata.id`.
- Each value is the compressed NPZ produced by `RefStructure.save_npz()`.
- The manifest is a MessagePack list of `Metadata.to_dict()` records. Its IDs,
  chains, residue counts, and interfaces must agree with the stored structures.
  Cluster annotations may be added to the manifest after structure processing.
- Predicted structures use `metadata.source = "pred"`. Prediction model and
  quality annotations are optional; the loader does not read source CSV files.
- Chain types, entity IDs, asymmetric chain IDs, symmetry IDs, atom order,
  ligand chemistry, and covalent connections are recorded during preprocessing.
  The loader preserves them, including repeated entities in symmetric complexes.
- The training pipeline still needs a CCD containing the required components.

Compact NPZ records containing only `sequence`/`coordinates`, or
`0.sequence`/`1.sequence`, are not supported. Rebuild those datasets as complete
`RefStructure` records and regenerate their manifests before using this loader.

Upstream CIF processing must produce a complete `RefStructure`. In particular,
custom ligands need chemical component/bond information or an explicit chemical
definition; a coordinate-only CIF is not a complete chemical description.

For a small dataset, existing `RefStructure` NPZ files can be packed as follows.
Use a fresh output directory and choose an LMDB map size appropriate to the data:

```python
from pathlib import Path

import lmdb
import msgpack

from kfold.data.types.structure import RefStructure

npz_dir = Path("/path/to/refstructure_npz")
output_dir = Path("/path/to/my_predictions")
output_dir.mkdir(parents=True, exist_ok=False)
manifest = []

with lmdb.open(str(output_dir / "structure.lmdb"), map_size=64 * 1024**3) as env:
    with env.begin(write=True) as txn:
        for path in sorted(npz_dir.rglob("*.npz")):
            structure = RefStructure.load_npz(path)
            structure.validate()
            if structure.metadata.source != "pred":
                raise ValueError(f"Expected predicted structure: {path}")
            if not txn.put(structure.id.encode("utf-8"), path.read_bytes(), overwrite=False):
                raise ValueError(f"Duplicate structure ID: {structure.id}")
            manifest.append(structure.metadata.to_dict())

with (output_dir / "manifest.msgpack").open("wb") as stream:
    msgpack.pack(manifest, stream)
```

## Training configuration

Add an entry under `train.data.train_datasets`. The `_yaml_` path below is
relative to a training config in `configs/train/`:

```yaml
train:
  data:
    train_datasets:
      - _yaml_: "dataset/default-train.yaml"
        type: distillation
        name: my_predictions
        data_path: /path/to/my_predictions
        weight: 1.0
        sampler: null
```

With `sampler: null`, sampling is uniform over structures, including structures
with a single chain. Configure cropping and augmentation for your task; neither the dataset
name nor the number of chains selects a different loader or augmentation policy.
Predicted labels never supervise the confidence head.

To use cluster-weighted chain/interface sampling, set the entry's sampler:

```yaml
sampler:
  _registry_: data_sampler
  _class_: ClusterSampler
  beta_chain: 0.5
  beta_interface: 1.0
  allow_redundant: false
```

Supply `cluster_id` for every chain and interface in the manifest.
The `beta_chain: 0.5` setting includes monomers. Setting `beta_chain: 0.0`
restricts sampling to interfaces, so structures without interfaces receive no
samples. Sampling policy is independent of the storage format.

## Apo and prior inputs

External apo conditioning is optional. Without monomer or multimer apo inputs,
protein apo coordinates and structure tokens remain masked; the target structure is
not automatically used as trunk apo input. No monomer-specific path substitutes
label coordinates or applies an additional prior perturbation.

To supply protein apo structures, use the shared layout:

```text
my_predictions/
  apo_lookup.msgpack
  apo_lmdb/protein/<source>.lmdb/
  apo_tok_lmdb/protein/<source>.lmdb/
  prior_lmdb/protein/<source>.lmdb/
```

The lookup maps structure ID to entity ID to a list of records containing
`chain_type: protein`, `source`, and `name`; `name` is the key in both the apo
and token LMDBs. A training record may include a `residue_map` for sequence
alignment. Repeated chains sharing an entity reuse its selected apo inputs.
The apo and token source LMDBs must exist for referenced sources.

Protein prior stacks are optional and use `<structure_id>_<entity_id>` keys.
Each source has its own file; the loader samples uniformly across all stored
ranks and sources.
If no stack is available, the common training pipeline supplies centered protein
label coordinates to the configured prior sampler. Nucleic-acid and ligand
priors follow that sampler's chain-type policy. Configure the prior sampler
through the data module; `DistillationDataset` does not generate external
predictions or update LMDBs during training.

### Protein multimer conditioning

RCSB and distillation datasets share protein multimer apo, prior, and token
loading. Enable them in a dataset entry with:

```yaml
prob_use_complex_apo: 0.5
prob_use_complex_prior: 0.5
```

The default training fragment sets both probabilities to 0.5. Supply these files:

```text
my_predictions/
  apo_multimer_lookup.msgpack
  apo_lmdb/protein-multimer/<source>.lmdb/
  apo_tok_lmdb/protein-multimer/<source>.lmdb/
  prior_lmdb/protein-multimer/<source>.lmdb/
```

The lookup maps each structure ID to a list of groups. For example:

```json
{
  "prediction_1": [
    {"name": "pair_1", "source": "predictor", "chain_type": "protein",
     "asym_ids": [1, 2], "apo_uid": 1}
  ]
}
```

`name` is the key in the apo, token, and prior LMDBs. Chain keys in their payloads
must match the target's integer `asym_id` values. Use the serialization helpers
in `kfold.training.dataset.utils.apo_io`: `pack_apo_multimer_record`,
`pack_apo_multimer_token_record`, and `pack_prior_multimer_stack_record`.

Sources with the same name, group ID, and chains are alternative apo inputs.
Selected sources occupy matching apo slots across member chains; prior stacks
select the same sample index across member chains. Give each physical group a
shared `apo_uid` that is distinct from unrelated chains/groups so downstream
processing preserves the group's relative placement.

Monomer lookup files are optional, including for datasets that supply only
multimer conditioning. Missing multimer lookups or disabled probabilities use
the common monomer/masked-input path. Missing multimer prior records use the
common prior fallback; missing multimer token files/records leave tokens masked.
Referenced multimer apo records must exist.

The current multimer apo selection policy requires distinct protein entities
within each group. Groups with repeated entities are skipped. Moving this
policy into the common training dataset does not add homomer apo support or
change validation dataset input handling.
