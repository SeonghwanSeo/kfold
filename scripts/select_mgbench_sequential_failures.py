"""Audit four existing top-1 pools and prepare a mechanistic failure cohort.

Run on a host mounting MGBench. No model/evaluator changes or job submissions.
Raw ligand assignments are diagnostic: their receptor context can include extra
reference chains. They are not silently substituted for standardized A+B PLI.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BLOCKS = [
    ("1-5", ""),
    ("6-10", "-seed6-10"),
    ("11-15", "-seed11-15"),
    ("16-20", "-seed16-20"),
]
PILOT = {"8OV6", "8BUA", "7EG1", "8PPZ", "8JYC", "8JYE", "8TBF", "8TBG", "8TBJ"}
OVERRIDES = {
    "8OV6": (
        "B",
        "BRD4-DCAF16",
        "BRD4-IBG1 first: intramolecular BD1/BD2 engagement "
        "remodels BRD4 for DCAF16 recruitment",
        "https://www.nature.com/articles/s41586-024-07089-6",
    ),
    "8PPZ": (
        "A",
        "FKBP12-mTOR",
        "FKBP12-compound 7 first; synthetic FKBP-binding glue, not rapamycin",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11796051/",
    ),
    "8JYC": (
        "B",
        "BTN2A1-BTN3A1",
        "BTN3A1-DMAPP first creates composite BTN2A1-binding interface; "
        "native oligomer caveat",
        "https://www.nature.com/articles/s41586-023-06525-3",
    ),
    "8JYE": (
        "B",
        "BTN2A1-BTN3A1",
        "BTN3A1-phosphoantigen first creates composite BTN2A1-binding interface; "
        "native oligomer caveat",
        "https://www.nature.com/articles/s41586-023-06525-3",
    ),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def success(v, threshold):
    return None if v is None else v > threshold


def joint(a, b):
    if a is False or b is False:
        return False
    return True if a is True and b is True else None


def csvwrite(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    annotations = {r["pdb"]: r for r in csv.DictReader(args.annotations.open())}
    assert len(annotations) == 88
    records, sources, snapshots, coverage = [], [], [], []
    for block, suffix in BLOCKS:
        run = "KFold-0720-68k-ssh-new" + suffix
        ev = args.benchmark / "eval" / run / "ost_top1"
        raw = ev / "ost/ost_raw_scores.csv"
        audit = ev / "prep/selection_audit.csv"
        assert (ev / "OST_FINISHED").exists(), ev
        for f in (raw, audit):
            sources.append({"path": str(f), "sha256": sha(f)})
        rows = list(csv.DictReader(raw.open()))
        keyed = {(x["pdb_id"].upper(), x["task"]): x for x in rows}
        assert len(rows) == len(keyed) == 264
        selected_rows = list(csv.DictReader(audit.open()))
        selected = {x["target"].upper(): x for x in selected_rows}
        assert len(selected_rows) == len(selected) == 88
        assert set(selected) == set(annotations)
        current = []
        for pdb in sorted(annotations):
            pp = keyed[pdb, "interface_protein_protein"]
            pl = keyed[pdb, "interface_protein_ligand"]
            s = selected[pdb]
            assert int(s["candidate_count"]) == 25, (run, pdb, s)
            assert Path(s["selected_cif"]).is_file()
            assert pp["sample_id"] == pl["sample_id"]
            predictions = list(
                (args.benchmark / "outputs" / run / pdb).glob(
                    f"{pdb}_seed-*/*_sample-*.cif"
                )
            )
            assert len(predictions) == 25, (run, pdb, len(predictions))
            ppfile, plfile = Path(pp["detail_json_path"]), Path(pl["detail_json_path"])
            dp, dl = json.loads(ppfile.read_text()), json.loads(plfile.read_text())
            assignments = [
                a
                for a in dl["lddt_pli"]["assigned_scores"]
                if a["model_ligand"].split(".")[0] == "C"
            ]
            assert len(assignments) <= 1, (run, pdb)
            assignment = assignments[0] if assignments else {}
            raw_pli = number(assignment.get("score"))
            rmsd_assignment = [
                a
                for a in dl["rmsd"]["assigned_scores"]
                if a["model_ligand"].split(".")[0] == "C"
            ]
            assert len(rmsd_assignment) <= 1
            dockq, pli = number(pp["dockq_score"]), number(pl["lddt_pli"])
            assert dockq is not None
            # Single mapped model A/B interface in the actual OST output.
            assert len(dp["dockq_interfaces"]) == 1
            assert set(dp["dockq_interfaces"][0][2:]) == {"A", "B"}
            assert math.isclose(dockq, dp["dockq"][0], abs_tol=1e-8)
            if pli is not None:
                assert raw_pli is not None and math.isclose(pli, raw_pli, abs_tol=1e-8)
            issue = (
                "numeric_csv"
                if pli is not None
                else (
                    "assignment_in_json_not_csv_reference_chain_filter"
                    if assignments
                    else "no_ligand_assignment"
                )
            )
            rec = dict(
                pdb=pdb,
                block=block,
                run=run,
                candidate_count=25,
                seed=s["seed"],
                sample=s["sample"],
                confidence=number(s["ranking_score"]),
                dockq=dockq,
                ppi_success=success(dockq, 0.23),
                lddt_pli_csv=pli,
                pli_success_csv=success(pli, 0.8),
                joint_success_csv=joint(success(dockq, 0.23), success(pli, 0.8)),
                ligand_rmsd_csv=number(pl["rmsd"]),
                lddt_pli_json_diagnostic=raw_pli,
                ligand_rmsd_json_diagnostic=number(rmsd_assignment[0]["score"])
                if rmsd_assignment
                else None,
                reference_ligand_json=assignment.get("reference_ligand", ""),
                ligand_coverage_json=assignment.get("coverage"),
                ligand_issue=issue,
                unassigned_reason=json.dumps(
                    dl["lddt_pli"].get("model_ligand_unassigned_reason", {})
                ),
                dockq_interfaces=json.dumps(dp["dockq_interfaces"]),
                selected_cif=s["selected_cif"],
                ost_version=dl.get("ost_version", dl.get("version", "")),
                ppi_json=str(ppfile),
                pli_json=str(plfile),
            )
            current.append(rec)
            snapshots.append(
                dict(
                    run=run,
                    pdb=pdb,
                    ppi=dp,
                    ligand=dl,
                    ppi_sha256=sha(ppfile),
                    ligand_sha256=sha(plfile),
                )
            )
        records.extend(current)
        coverage.append(
            dict(
                block=block,
                systems=88,
                candidates=2200,
                ppi_numeric=88,
                ppi_pass=sum(x["ppi_success"] for x in current),
                pli_numeric_csv=sum(x["lddt_pli_csv"] is not None for x in current),
                pli_numeric_json_diagnostic=sum(
                    x["lddt_pli_json_diagnostic"] is not None for x in current
                ),
                issues=dict(Counter(x["ligand_issue"] for x in current)),
            )
        )
    summary, candidates, plans = [], [], []
    for pdb, ann in sorted(annotations.items()):
        group = [x for x in records if x["pdb"] == pdb]
        assert len(group) == 4
        first, family, reason, url = OVERRIDES.get(
            pdb,
            (
                ann["recommended_first_chain"],
                ann["family"],
                ann["rationale"],
                ann["source_url"],
            ),
        )
        repeat = all(x["ppi_success"] is False for x in group)
        row = dict(
            pdb=pdb,
            protein_A=ann["protein_A"],
            protein_B=ann["protein_B"],
            ligand_C=ann["ligand_C"],
            official_class=ann["official_class"],
            family=family,
            **{f"dockq_seed{x['block']}": x["dockq"] for x in group},
            dockq_min=min(x["dockq"] for x in group),
            dockq_max=max(x["dockq"] for x in group),
            ppi_failed_blocks=sum(x["ppi_success"] is False for x in group),
            pli_numeric_blocks_csv=sum(x["lddt_pli_csv"] is not None for x in group),
            pli_numeric_blocks_json=sum(
                x["lddt_pli_json_diagnostic"] is not None for x in group
            ),
            repeated_ppi_failure=repeat,
            cohort=("pilot9" if pdb in PILOT else "expansion12")
            if repeat and first
            else "not_selected",
            first_chain=first if repeat else "",
            first_protein=ann.get(f"protein_{first}", "") if repeat else "",
            rationale=reason,
            source_url=url,
        )
        if pdb == "8VOJ":
            row["rationale"] = (
                "Defer: UM171 binary binding not detected; "
                "native KBTBD4 dimer, CoREST and InsP6 context important"
            )
            row["source_url"] = "https://www.nature.com/articles/s41586-024-08532-4"
        if pdb == "7TE8":
            row["rationale"] = (
                "Defer: CA14 homomer; no protein-selective binary-first "
                "rationale established in this audit"
            )
        summary.append(row)
        if not repeat or not first:
            continue
        candidates.append(row)
        src = Path(ann["source_input"])
        assert sha(src) == ann["source_sha256"], src
        q = json.loads(src.read_text())
        assert set(q) == {"name", "sequences"} and q["name"] == pdb
        for item in q["sequences"]:
            if "protein" in item:
                apo = item["protein"]["apo"]
                for path in [apo] if isinstance(apo, str) else apo:
                    assert Path(path).is_file(), path
        for variant, chain in [
            ("direct", None),
            ("mechanism_first", first),
            ("reverse_order_control", "B" if first == "A" else "A"),
        ]:
            query = copy.deepcopy(q)
            if chain:
                query["assembly"] = {
                    "stages": [{"id": f"bind_{chain}_C", "chains": [chain, "C"]}],
                    "selection": "confidence_top1",
                }
            assert {k: v for k, v in query.items() if k != "assembly"} == q
            plans.append((row["cohort"], variant, pdb, query))
    assert len(candidates) == 21 and sum(x["cohort"] == "pilot9" for x in candidates) == 9
    assert sum(x["repeated_ppi_failure"] for x in summary) == 23
    args.out.mkdir(parents=True)
    csvwrite(args.out / "baseline_four_pools_352rows.csv", records)
    csvwrite(args.out / "all88_system_summary.csv", summary)
    csvwrite(
        args.out / "repeated_ppi_failures23.csv",
        [x for x in summary if x["repeated_ppi_failure"]],
    )
    csvwrite(args.out / "sequential_candidates21.csv", candidates)
    for cohort, variant, pdb, query in plans:
        path = args.out / "inputs" / cohort / variant / "queries" / f"{pdb}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(query, indent=2) + "\n")
    (args.out / "ost_detail_snapshot.json").write_text(json.dumps(snapshots) + "\n")
    provenance = dict(
        coverage=coverage,
        source_csv_hashes=sources,
        annotations=str(args.annotations),
        annotations_sha256=sha(args.annotations),
        script_sha256=sha(Path(__file__)),
        thresholds={"ppi": "DockQ > 0.23", "pli": "lDDT-PLI > 0.8"},
        selection=(
            "4 independent confidence top-1 selections, "
            "each from 25 candidates; NOT oracle100"
        ),
        warnings=[
            "JSON ligand scores are diagnostic, not standardized two-protein-only PLI",
            "Mechanistic anchors are hypotheses, not proof of unique kinetic order",
            "Original apo/SMILES preserved; no holo coordinates enter inputs",
            "No inference, evaluator modification or Slurm submission performed",
            "Native parser and neural load/forward checks are separate steps",
        ],
        counts={
            "baseline_rows": len(records),
            "repeat_ppi_failures": 23,
            "candidates": 21,
            "pilot": 9,
            "expansion": 12,
        },
    )
    (args.out / "audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
