import pytest

from kfold.training.metrics.structure_metrics import (
    aggregate_validation_metrics,
    extract_validation_metrics,
    main_metric_names,
)


def _summary(homo_lddt: float, hetero_lddt: float, rmsd: float) -> dict:
    def interface(lddt: float, entity_1: int, entity_2: int) -> dict:
        return {
            "is_low_homology": True,
            "metrics": {"lddt": lddt},
            "type_1": "protein",
            "type_2": "protein",
            "subtype_1": "protein",
            "subtype_2": "protein",
            "entity_id_1": entity_1,
            "entity_id_2": entity_2,
        }

    return {
        "id": "test",
        "metrics": {"rmsd": rmsd, "lddt": (homo_lddt + hetero_lddt) / 2},
        "chains": {},
        "interfaces": {
            "A:B": interface(homo_lddt, 1, 1),
            "A:C": interface(hetero_lddt, 1, 2),
        },
    }


def test_pp_interface_metrics_keep_combined_and_split_homo_hetero() -> None:
    metrics = extract_validation_metrics(_summary(0.2, 0.8, 3.0))

    assert metrics["interface/lddt-protein_protein"] == pytest.approx(0.5)
    assert metrics["special/interface/lddt_protein_protein_homo"] == 0.2
    assert metrics["special/interface/lddt_protein_protein_hetero"] == 0.8


def test_pp_interface_splits_are_registered_for_top1_and_top5() -> None:
    homo_key = "special/interface/lddt_protein_protein_homo"
    hetero_key = "special/interface/lddt_protein_protein_hetero"
    assert homo_key in main_metric_names
    assert hetero_key in main_metric_names

    aggregated = aggregate_validation_metrics(
        [_summary(0.2, 0.8, 3.0), _summary(0.7, 0.4, 2.0)],
        top1_idx=0,
    )
    assert aggregated["top1"][homo_key] == 0.2
    assert aggregated["top1"][hetero_key] == 0.8
    assert aggregated["top5"][homo_key] == 0.7
    assert aggregated["top5"][hetero_key] == 0.8
