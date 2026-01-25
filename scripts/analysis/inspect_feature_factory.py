from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from rdkit import Chem

from kfold.data.utils.ligand_interactions import _get_feature_factory

FeatureRow = tuple[str, str, tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RDKit feature factory output for SMILES strings."
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        help="One or more SMILES strings to inspect.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        help="Path to a text file containing one SMILES per line.",
    )
    parser.add_argument(
        "--add-hs",
        action="store_true",
        help="Add explicit hydrogens before feature detection.",
    )
    return parser.parse_args()


def read_smiles_file(path: Path) -> list[str]:
    smiles_list: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smiles = stripped.split()[0]
            smiles_list.append(smiles)
    return smiles_list


def gather_smiles(args: argparse.Namespace) -> list[str]:
    smiles_list: list[str] = []
    if args.smiles:
        smiles_list.extend(args.smiles)
    if args.smiles_file:
        smiles_list.extend(read_smiles_file(args.smiles_file))
    return smiles_list


def build_mol(smiles: str, add_hs: bool) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol


def extract_feature_rows(features: Iterable[object]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for feature in features:
        family = feature.GetFamily()
        feature_type = feature.GetType()
        atom_ids = tuple(int(atom_id) for atom_id in feature.GetAtomIds())
        rows.append((family, feature_type, atom_ids))
    return rows


def features_per_atom(num_atoms: int, feature_rows: list[FeatureRow]) -> list[list[str]]:
    per_atom: list[list[str]] = [[] for _ in range(num_atoms)]
    for family, _, atom_ids in feature_rows:
        for atom_id in atom_ids:
            per_atom[atom_id].append(family)

    for atom_id, families in enumerate(per_atom):
        seen: set[str] = set()
        unique: list[str] = []
        for family in families:
            if family in seen:
                continue
            seen.add(family)
            unique.append(family)
        per_atom[atom_id] = unique

    return per_atom


def print_results(smiles: str, mol: Chem.Mol, feature_rows: list[FeatureRow]) -> None:
    per_atom = features_per_atom(mol.GetNumAtoms(), feature_rows)

    print(f"SMILES: {smiles}")
    print(f"Num atoms: {mol.GetNumAtoms()}")
    print("Atoms:")
    for atom in mol.GetAtoms():
        print(f"  {atom.GetIdx():>3} {atom.GetSymbol()}")

    print("Features:")
    if feature_rows:
        for idx, (family, feature_type, atom_ids) in enumerate(feature_rows, start=1):
            atom_ids_str = ", ".join(str(atom_id) for atom_id in atom_ids)
            print(
                f"  {idx:>3}. Family={family} Type={feature_type} "
                f"AtomIds=({atom_ids_str})"
            )
    else:
        print("  (none)")

    print("Per-atom families:")
    for atom in mol.GetAtoms():
        families = per_atom[atom.GetIdx()]
        families_str = ", ".join(families) if families else "-"
        print(f"  {atom.GetIdx():>3} {atom.GetSymbol()}: {families_str}")
    print()


def main() -> None:
    args = parse_args()
    smiles_list = gather_smiles(args)
    if not smiles_list:
        raise SystemExit("Provide --smiles or --smiles-file.")

    factory = _get_feature_factory()
    print("DEBUG] factory:", factory)
    for smiles in smiles_list:
        mol = build_mol(smiles, args.add_hs)
        features = factory.GetFeaturesForMol(mol)
        feature_rows = extract_feature_rows(features)
        print_results(smiles, mol, feature_rows)


if __name__ == "__main__":
    main()
