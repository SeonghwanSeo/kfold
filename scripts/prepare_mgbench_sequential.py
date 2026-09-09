"""Prepare, not submit, lossless MGBench sequential-order experiments.

GT files are used ONLY for sequence-to-entity annotations, never coordinates,
contacts, ranking or priors. Mechanistic choices are hypotheses, not kinetics.
"""

import argparse
import copy
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import gemmi


def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def official_rows(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        strings = [
            "".join(n.itertext()) for n in ET.fromstring(z.read("xl/sharedStrings.xml"))
        ]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall(
            ".//m:row", ns
        ):
            cells = {}
            for cell in row:
                value = cell.find("m:v", ns)
                value = value.text if value is not None else "".join(cell.itertext())
                if cell.get("t") == "s":
                    value = strings[int(value)]
                cells["".join(c for c in cell.get("r") if c.isalpha())] = value
            rows.append(cells)
    result = {}
    for row in rows[2:]:
        pdb = row.get("A", "").upper().strip()
        if len(pdb) == 4:
            if pdb in result:
                raise ValueError(f"Duplicate official annotation: {pdb}")
            result[pdb] = row
    return result


def recommendation(names):
    for chain, name in names.items():
        if "Cyclin-dependent kinase 12" in name:
            return (
                chain,
                "CDK12-DDB1",
                "CDK12 pocket; degraded substrate cyclin K is not CDK12",
                "https://www.nature.com/articles/s41589-023-01409-z",
            )
        if "cereblon" in name.lower():
            return (
                chain,
                "CRBN",
                "CRBN pocket; family-level anchor hypothesis, not measured kinetic order",
                "https://www.nature.com/articles/nature13527",
            )
        if "Peptidyl-prolyl cis-trans isomerase A" == name:
            return (
                chain,
                "RAS-CYPA",
                "CYPA binary-complex mechanism; "
                "family-level transfer for related compounds",
                "https://www.nature.com/articles/s41586-024-07205-6",
            )
        if "phosphodiesterase A" in name:
            return (
                chain,
                "PDE3A-SLFN12",
                "DNMDP binds PDE3A catalytic pocket",
                "https://www.nature.com/articles/s41467-021-26546-8",
            )
    family = "14-3-3-peptide" if any("14-3-3" in n for n in names.values()) else "other"
    return (
        "",
        family,
        "No unique first protein assigned; test both orders as algorithmic controls",
        "",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    official = official_rows(args.metadata)
    source = args.benchmark / "inputs/KFold/seed1-5_sample5/queries"
    paths = sorted(source.glob("*.json"))
    assert len(paths) == 88, len(paths)
    prepared, manifest, apo_files = [], [], set()
    for path in paths:
        q = json.loads(path.read_text())
        assert q["name"] == path.stem and set(q) == {"name", "sequences"}, path
        proteins = {
            x["protein"]["id"]: x["protein"] for x in q["sequences"] if "protein" in x
        }
        ligands = [x["ligand"] for x in q["sequences"] if "ligand" in x]
        assert set(proteins) == {"A", "B"} and len(q["sequences"]) == 3, path
        assert len(ligands) == 1 and ligands[0]["id"] == "C", path
        gt = args.benchmark / "artifacts/GT" / f"{path.stem}.cif"
        block = gemmi.cif.read(str(gt)).sole_block()
        descriptions = {
            e: gemmi.cif.as_string(n)
            for e, n in zip(
                block.find_values("_entity.id"),
                block.find_values("_entity.pdbx_description"),
                strict=True,
            )
        }
        entities = list(
            zip(
                block.find_values("_entity_poly.entity_id"),
                block.find_values("_entity_poly.pdbx_seq_one_letter_code_can"),
                strict=True,
            )
        )
        names = {}
        for chain, protein in proteins.items():
            matches = {
                descriptions[e]
                for e, seq in entities
                if protein["sequence"] == "".join(gemmi.cif.as_string(seq).split())
            }
            assert len(matches) == 1, (path, chain, matches)
            names[chain] = matches.pop()
            apo = protein["apo"]
            for filename in [apo] if isinstance(apo, str) else apo:
                assert Path(filename).is_file(), filename
                apo_files.add(Path(filename))
        first, family, rationale, url = recommendation(names)
        row = official[path.stem]
        manifest.append(
            dict(
                pdb=path.stem,
                protein_A=names["A"],
                protein_B=names["B"],
                ligand_C=row["B"],
                official_class=row["D"],
                family=family,
                recommended_first_chain=first,
                status="anchor_hypothesis" if first else "unassigned",
                rationale=rationale,
                source_url=url,
                source_input=str(path),
                source_sha256=sha(path),
                annotation_cif_sha256=sha(gt),
            )
        )
        variants = {"direct": None, "A_first": "A", "B_first": "B"}
        if first:
            variants["mechanism_first"] = first
        for variant, chain in variants.items():
            value = copy.deepcopy(q)
            if chain:
                value["assembly"] = {
                    "stages": [{"id": f"bind_{chain}_C", "chains": [chain, "C"]}],
                    "selection": "confidence_top1",
                }
            assert {k: v for k, v in value.items() if k != "assembly"} == q
            prepared.append((variant, path.name, value))
    # No output is created until all 88 inputs and their dependencies pass.
    for variant, filename, value in prepared:
        folder = args.out / variant / "queries"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text(json.dumps(value, indent=2) + "\n")
    with (args.out / "order_manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = dict(
        systems=88,
        variants=dict(Counter(v for v, _, _ in prepared)),
        families=dict(Counter(r["family"] for r in manifest)),
        official_classes=dict(Counter(r["official_class"] for r in manifest)),
        seeds=[1, 2, 3, 4, 5],
        samples_per_seed=5,
        recycles=10,
        diffusion_steps=200,
        metadata=str(args.metadata),
        metadata_sha256=sha(args.metadata),
        apo_hashes={str(p): sha(p) for p in sorted(apo_files)},
        validation=(
            "JSON, exact sequence annotation, apo existence, lossless source "
            "comparison; no model load or GPU forward"
        ),
        warning=(
            "mechanism_first is a partial, literature-informed subset; do not "
            "silently treat it as MGBench-88. No ground-truth geometry enters "
            "inference. No jobs submitted."
        ),
    )
    (args.out / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "apo_hashes"}, indent=2))


if __name__ == "__main__":
    main()
