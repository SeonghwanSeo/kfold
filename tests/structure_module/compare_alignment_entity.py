from collections import Counter
from pathlib import Path

import torch

import kfold.constants as C
import kfold.model.modules as submodules
from kfold.config import load_config
from kfold.training.folding.dataset.datamodule import TrainingDataModule
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.geometry.rigid_align import weighted_rigid_align
from kfold.utils.registry import Registry

TEST_CONFIG_PATH = Path("configs/train-esm2-ecsi-mini-pairmixer.yaml")
SAVE_PATH = Path("./tmp/align_compare")


def align_all_entities(
    apo_coords: torch.Tensor,
    label_coords: torch.Tensor,
    apo_mask: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    # apo_coords/label_coords: [B, N, La, 3]
    B, N, L = apo_coords.shape[:3]
    apo_mask_exp = apo_mask[:, None, :].expand(B, N, L)
    label_mask_exp = label_mask[:, None, :].expand(B, N, L)
    align_mask = apo_mask_exp & label_mask_exp

    apo_centered = do_centering(apo_coords, align_mask, mask_to_zero=False)
    aligned = weighted_rigid_align(
        coords=apo_centered,
        target=label_coords,
        weights=align_mask.to(dtype=apo_coords.dtype),
        mask=align_mask,
    )
    return aligned * apo_mask_exp[..., None]


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.train_batch_size = 1
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0
    global_config.train.data.safe_load = False
    global_config.train.data.apo_perturbation_args.metric_lmdb_path = (
        "/cache/wykim_lab/kfold_data/apo_metrics_esmfold.lmdb"
    )

    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    dataloader = data_module.val_dataloader()

    score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
        global_config.model.score_model
    )
    structure_module: submodules.structure_module.BaseStructureModule = (
        Registry.instantiate(
            global_config.model.structure_module, score_model=score_model
        )
    )

    for data in iter(dataloader):
        f_input, full_dict_list = data
        print(f_input.chain.chain_type.tolist()[0])
        protein_ge_2 = (
            Counter(f_input.chain.chain_type.tolist()[0])[C.ChainType.PROTEIN.value] >= 2
        )
        not_too_big = f_input.num_tokens < 600
        print(
            full_dict_list[0]["id"],
            f_input.num_chains,
            f_input.num_tokens,
            Counter(f_input.chain.chain_type.tolist()[0]),
        )
        if f_input.num_chains > 4 and protein_ge_2 and not_too_big:
            print(f_input.chain.chain_type)
            print("FOUND!!")
            print(f_input)
            break
    else:
        exit()

    full_dict = full_dict_list[0]
    name = full_dict["id"]
    struct = full_dict["structure"]
    print(full_dict)

    num_samples = 1
    label_coords = structure_module.sample_holo(f_input, num_samples)
    apo_raw = structure_module.sample_apo(
        f_input, num_diffusion_samples=num_samples, random_augment=False
    )

    # (A) entity-selection align (current behavior)
    structure_module.alignment_entity_strategy = "largest"
    apo_entity = structure_module.align_apo_to_label_by_entity_selection(
        apo_raw, label_coords, f_input
    )

    # (B) all-entity align (baseline)
    apo_all = align_all_entities(
        apo_raw, label_coords, f_input.atom.apo_mask, f_input.atom.resolved_mask
    )

    # Optional: random_non_ligand variant
    structure_module.alignment_entity_strategy = "random_non_ligand"
    apo_entity_random = structure_module.align_apo_to_label_by_entity_selection(
        apo_raw, label_coords, f_input
    )

    # CIF output (same topology, different coords)
    struct.replace_atom_coords(label_coords[0].numpy()).to_mmcif(
        SAVE_PATH / f"{name}-holo.cif", is_predicted=False
    )
    struct.replace_atom_coords(apo_raw[0].numpy()).to_mmcif(
        SAVE_PATH / f"{name}-apo_raw.cif", is_predicted=False
    )
    struct.replace_atom_coords(apo_entity[0].numpy()).to_mmcif(
        SAVE_PATH / f"{name}-apo_entity_largest.cif", is_predicted=False
    )
    struct.replace_atom_coords(apo_all[0].numpy()).to_mmcif(
        SAVE_PATH / f"{name}-apo_all_entities.cif", is_predicted=False
    )
    struct.replace_atom_coords(apo_entity_random[0].numpy()).to_mmcif(
        SAVE_PATH / f"{name}-apo_entity_random.cif", is_predicted=False
    )


if __name__ == "__main__":
    main()
