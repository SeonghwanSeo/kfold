import pytest

from kfold.training.metrics.structure_metrics import (
    aggregate_validation_metrics,
    extract_validation_metrics,
    main_metric_names,
)


def _summary(homo_lddt: float, hetero_lddt: float, rmsd: float) -> dict:
    def interface(lddt: float, descriptor: str) -> dict:
        return {
            "is_low_homology": True,
            "metrics": {"lddt": lddt},
            "type_1": "protein",
            "type_2": "protein",
            "subtype_1": "protein",
            "subtype_2": "protein",
            "descriptors": [descriptor],
        }

    return {
        "id": "test",
        "metrics": {"rmsd": rmsd, "lddt": (homo_lddt + hetero_lddt) / 2},
        "chains": {},
        "interfaces": {
            "A:B": interface(homo_lddt, "homo"),
            "A:C": interface(hetero_lddt, "hetero"),
        },
    }


def test_pp_interface_metrics_keep_combined_and_split_homo_hetero() -> None:
    metrics = extract_validation_metrics(_summary(0.2, 0.8, 3.0))

    assert metrics["interface/lddt-protein_protein"] == pytest.approx(0.5)
    assert metrics["special/interface/lddt_protein_protein_homo"] == 0.2
    assert metrics["special/interface/lddt_protein_protein_hetero"] == 0.8


def test_pp_interface_splits_are_not_inferred_without_descriptors() -> None:
    summary = _summary(0.2, 0.8, 3.0)
    for interface in summary["interfaces"].values():
        interface["descriptors"] = []

    metrics = extract_validation_metrics(summary)

    assert "special/interface/lddt_protein_protein_homo" not in metrics
    assert "special/interface/lddt_protein_protein_hetero" not in metrics


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


def _summary_with_antibody_antigen(
    regular_hetero_lddt: float,
    antibody_antigen_lddt: float,
    rmsd: float,
) -> dict:
    def interface(lddt: float, descriptors: list[str]) -> dict:
        return {
            "is_low_homology": True,
            "metrics": {"lddt": lddt},
            "type_1": "protein",
            "type_2": "protein",
            "subtype_1": "protein",
            "subtype_2": "protein",
            "descriptors": descriptors,
        }

    return {
        "id": "test",
        "metrics": {
            "rmsd": rmsd,
            "lddt": (regular_hetero_lddt + antibody_antigen_lddt) / 2,
        },
        "chains": {},
        "interfaces": {
            "A:B": interface(regular_hetero_lddt, ["hetero"]),
            "A:C": interface(
                antibody_antigen_lddt,
                ["hetero", "antibody_antigen"],
            ),
        },
    }


def test_antibody_antigen_metric_overlaps_existing_hetero_metric() -> None:
    antibody_antigen_key = "special/interface/lddt_protein_protein_antibody_antigen"
    hetero_key = "special/interface/lddt_protein_protein_hetero"

    metrics = extract_validation_metrics(_summary_with_antibody_antigen(0.2, 0.8, 3.0))

    assert metrics[hetero_key] == pytest.approx(0.5)
    assert metrics[antibody_antigen_key] == 0.8
    assert antibody_antigen_key in main_metric_names

    aggregated = aggregate_validation_metrics(
        [
            _summary_with_antibody_antigen(0.2, 0.8, 3.0),
            _summary_with_antibody_antigen(0.6, 0.4, 2.0),
        ],
        top1_idx=0,
    )
    assert aggregated["top1"][antibody_antigen_key] == 0.8
    assert aggregated["top5"][antibody_antigen_key] == 0.8
