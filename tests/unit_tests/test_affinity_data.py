from dataclasses import replace
from io import StringIO

import pytest

from kfold.training.affinity.corpus import (
    bindingdb_residual_from_sair,
    iter_bindingdb_rows,
    normalize_bindingdb_row,
    normalize_sair_record,
    parse_exact_numeric,
)
from kfold.training.affinity.data import (
    Endpoint,
    assign_assay_split,
    p_activity_scale,
    protein_key,
    remove_excluded_proteins,
    system_key,
)
from kfold.training.affinity.sequence import normalize_protein_sequence


def _bindingdb_row(**overrides: str) -> dict[str, str]:
    row = {
        "UniProt (SwissProt) Primary ID of Target Chain": "P00001",
        "BindingDB Target Chain Sequence": "ACDEFG",
        "Ligand SMILES": "CCO",
        "BindingDB Reactant_set_id": "123",
        "Assay ID": "assay-1",
        "Ki (nM)": "10",
        "Kd (nM)": "",
        "IC50 (nM)": "",
        "EC50 (nM)": "",
    }
    row.update(overrides)
    return row


def test_bindingdb_endpoint_conversion_and_qualifier_rejection() -> None:
    records = normalize_bindingdb_row(
        _bindingdb_row(**{"Ki (nM)": "= 10", "IC50 (nM)": "50"})
    )
    assert {record.endpoint for record in records} == {Endpoint.KI, Endpoint.IC50}
    assert {round(record.p_activity, 4) for record in records} == {8.0, 7.301}
    assert parse_exact_numeric("<10") is None
    assert parse_exact_numeric(">10") is None
    assert parse_exact_numeric("~10") is None
    assert parse_exact_numeric("10-20") is None
    assert parse_exact_numeric("10") == 10.0


def test_bindingdb_residual_removes_sair_represented_measurement() -> None:
    bindingdb = normalize_bindingdb_row(_bindingdb_row())
    sair_proxy = replace(bindingdb[0], origin="SAIR", source="BindingDB")
    assert bindingdb_residual_from_sair([sair_proxy], bindingdb) == []


def test_bindingdb_doi_is_provenance_not_a_ranking_assay() -> None:
    first = normalize_bindingdb_row(
        _bindingdb_row(
            **{
                "Assay ID": "",
                "BindingDB Reactant_set_id": "first",
                "DOI": "10.1000/example",
            }
        )
    )[0]
    second = normalize_bindingdb_row(
        _bindingdb_row(
            **{
                "Assay ID": "",
                "BindingDB Reactant_set_id": "second",
                "DOI": "10.1000/example",
            }
        )
    )[0]
    assert first.source_document_id == second.source_document_id == "10.1000/example"
    assert first.assay_key != second.assay_key
    assert "10.1000/example" not in first.assay_key


def test_unannotated_sair_rows_without_source_ids_are_singletons() -> None:
    system = {
        "sair_entry_id": 1,
        "protein_uniprot": "P00001",
        "sequence": "ACDE",
        "canonical_smiles": "CCO",
    }
    first = normalize_sair_record(
        system,
        {
            "affinity_label_index": 1,
            "affinity_label_uid": "first",
            "source": "BindingDB",
            "source_smiles": "CCO",
            "pIC50": "7.0",
        },
    )
    second = normalize_sair_record(
        system,
        {
            "affinity_label_index": 2,
            "affinity_label_uid": "second",
            "source": "BindingDB",
            "source_smiles": "CCO",
            "pIC50": "7.1",
        },
    )
    assert first is not None and second is not None
    assert first.assay_key != second.assay_key


def test_integerized_p_scale_uses_decimal_half_up_policy() -> None:
    assert p_activity_scale(7.30005) == 73001


def test_sequence_key_contract_normalizes_uncommon_residues() -> None:
    assert normalize_protein_sequence("AU1C\n") == "ACXC"
    assert system_key(sequence="AU1C", canonical_smiles="CCO") == system_key(
        sequence="ACXC", canonical_smiles="CCO"
    )
    assert protein_key(protein_uniprot="P1", sequence="AU1C") == protein_key(
        protein_uniprot="P1", sequence="ACXC"
    )


def test_fep_exclusion_and_deterministic_compound_split() -> None:
    record = normalize_bindingdb_row(_bindingdb_row())[0]
    second = replace(record, record_id="bindingdb:124:Ki")
    assigned = assign_assay_split([record, second], seed=7)
    assert assigned[0].split == assigned[1].split
    assert remove_excluded_proteins(assigned, {"P00001"}) == []
    assert remove_excluded_proteins(assigned, {"other"}) == assigned


def test_bindingdb_rejects_non_positive_endpoint() -> None:
    assert normalize_bindingdb_row(_bindingdb_row(**{"Ki (nM)": "0"})) == []
    with pytest.raises(ValueError):
        # Endpoint enum validation stays strict for any caller constructing
        # records outside the tabular source adapters.
        Endpoint("not-an-endpoint")


def test_bindingdb_duplicate_header_keeps_first_sequence_and_uniprot_column() -> None:
    text = (
        "Ligand SMILES\tBindingDB Target Chain  Sequence\t"
        "UniProt (SwissProt) Primary ID of Target Chain\t"
        "BindingDB Target Chain Sequence\t"
        "UniProt (SwissProt) Primary ID of Target Chain\n"
        "CCO\tACDE\tP00001\n"
    )
    row = next(iter_bindingdb_rows(StringIO(text)))
    assert row["BindingDB Target Chain  Sequence"] == "ACDE"
    assert row["UniProt (SwissProt) Primary ID of Target Chain"] == "P00001"
