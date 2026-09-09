# MGBench-88 sequential experiment preparation — 2026-09-09

요청 대화 `01a043cd-7036-7963-877f-82637b36ffed`의 로컬 세션 기록과
현재 GPU01 canonical MGBench 입력 및 실행 기록을 확인했다.
추론은 아직 제출하지 않았다. 입력 준비와 native parser/assembly plan 검증만 완료했다.

## 입력

공용 실험 입력 root:
`/mnt/parallel_storage/wykim_lab/share/kfold_special_benchmarks/MGBench/inputs/KFold/sequential_0720_68k_20260909`

| 하위 폴더 | 시스템 수 | 의미 |
|---|---:|---|
| `direct/queries` | 88 | 기존 canonical JSON의 무손실 복사 |
| `A_first/queries` | 88 | A–C 예측 → 전체 A–B–C |
| `B_first/queries` | 88 | B–C 예측 → 전체 A–B–C |
| `mechanism_first/queries` | 43 | 아래의 문헌 기반 anchor 방향; 전체 88개가 아님 |

최종 단계는 runner가 자동 추가한다. 중간 단계는 모든 seed/sample에서
confidence top-1 하나를 선택하고 다음 단계의 한 prior object로 전달한다.
전체 trunk를 다시 실행하며 중간 복합체를 denoising 중 rigid하게 고정하지 않는다.

예: `8BU1`은 A=DDB1, B=CDK12, C=glue이므로 권장 입력은 다음 block을 추가한다.

```yaml
assembly:
  stages:
    - id: bind_B_C
      chains: [B, C]
  selection: confidence_top1
```

기존 `sequences`, SMILES, apo 경로를 그대로 보존했다. 모든 입력에서 단백질
A/B와 ligand C를 확인했다. 원본 CIF에서는 entity 이름과 canonical sequence만
사용해 입력 chain의 이름을 정확히 매핑했다. GT 좌표·contact·성능은 순서 결정이나
입력 prior에 사용하지 않았다. apo 파일 존재와 SHA256, 원본 JSON 및 annotation
CIF/Excel SHA256은 preparation.json / order_manifest.csv에 기록했다.

## 분류와 결합 순서

공식 `data/data.xlsx`와 88 PDB를 조인한 분류:
degrader 32 / heterodimerization-nondegrader 55 / homodimerization 1.
따라서 전체 MGBench에 POI/ligase 구분을 강제할 수 없다.

| 입력 단백질 계열 | 수 | 권장 첫 binary | 근거 범위 |
|---|---:|---|---|
| CDK12–DDB1 | 20 | CDK12–glue | CDK12 pocket에 결합 후 DDB1 recruitment; 실제 분해 기질 cyclin K와 CDK12를 혼동하지 않음 |
| CRBN | 7 | CRBN–glue | CRBN pocket 기반 계열 수준 가설; 개별 화합물의 시간 순서 검증 아님 |
| RAS–CYPA | 15 | CYPA–glue | RMC-7977 binary-complex 기전 및 유사 계열에 대한 가설 |
| PDE3A–SLFN12 | 1 | PDE3A–glue | DNMDP의 PDE3A pocket 결합 |
| 14-3-3–peptide | 30 | 미지정 | 협동적 복합체에서 binary-first 유리성을 단정하지 않음 |
| 기타 | 15 | 미지정 | 개별 화합물 기전 추가 검토 필요 |

위의 권장 방향은 실험 설계용 anchor 가설이다. 구조적 binding pocket이
알려져 있다는 것과 생체 내에서 유일한 결합 시간 순서가 입증되었다는 것은 다르다.
특히 중간 binary가 불안정할 수 있어 순차 추론의 개선은 보장되지 않는다.
미지정 45개도 A_first/B_first 입력을 모두 만들어두었다. 어느 방향의 평가값이
좋은지 보고 사후에 고르는 것은 confidence-top-1 결과가 아니라 order oracle이므로
그렇게 합쳐 보고하지 않는다. 방향별 전체 88개 비교와 사전 지정 43개 비교를 구분한다.

Primary sources:

- [MGBench official metadata](https://github.com/yiyanliao/MGBench)
- [CDK12–DDB1 glue design](https://www.nature.com/articles/s41589-023-01409-z)
- [CRBN drug-binding pocket](https://www.nature.com/articles/nature13527)
- [RAS–CYPA, including 8TBF–8TBN](https://www.nature.com/articles/s41586-024-07205-6)
- [PDE3A–SLFN12, including 7EG1](https://www.nature.com/articles/s41467-021-26546-8)

## Checkpoint / runtime

- EMA: `/mnt/parallel_storage/share/kfold-benchmarks/tools/kfold_inference/resources/kfold-68K/epoch0135_step00068000_ema.pth`
- Config: `/mnt/parallel_storage/share/kfold-benchmarks/tools/kfold_inference/resources/kfold-68K/train_config.yaml`
- CCD: `/mnt/parallel_storage/share/kfold-benchmarks/tools/kfold_inference/resources/kfold-68K/ccd-train-v260701-af3.pkl`
- Historical source: `84363cfd538705184dc655dab596aaa900293cb0`
- Sequential worktree: `/home/icl_hwkim/kfold/_worktrees/sequential-complex-prior`
- Sequential base main: `5705156d34e87bdf302ee3aa9aaf4d0a385ebaad`
- Python: `/home/icl_hwkim/miniforge3/envs/kfold/bin/python`
- Each stage: seeds 1–5 × samples 5, recycles 10, diffusion steps **200**.

**Validation status:** A_first 88, B_first 88, mechanism_first 43 all passed actual
KFold native parsing and sequential plan validation with the 68K CCD.
This is not a neural checkpoint strict-load or GPU forward test. These remain
required before bulk submission. Do not silently disable strict loading or use
the runner's default 100 steps. Library threads must be capped; GPU runs use Slurm.

For causal comparison, the direct arm should also be run with the sequential
branch's code/config. A historical direct result from a different code revision
is a reference, not an isolated estimate of the effect of sequential inference.
Each sequential experiment adds a 25-candidate binary stage; equal final sample
counts do not imply equal total compute.

Reproduction script: `scripts/prepare_mgbench_sequential.py` (refuses overwrites).
Local per-system table: `docs/mgbench_sequential_order_manifest.csv`.
