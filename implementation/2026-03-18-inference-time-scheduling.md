## Scope

Update `main` with the ECSI inference time-scheduling change only.

Explicitly excluded from this update:

- staging and POC changes from the working branch
- distogram/oracle prior experiments
- schedule analysis and plotting scripts

## Files Changed

- `configs/model/module/structure_module/ecsi.yaml`
- `src/kfold/model/modules/structure_module/kfold_ecsi.py`

## Implementation Notes

The default ECSI sampling schedule in the config is switched from
`piecewise_power` to `phase_power`.

The new default schedule exposes three regions:

- early churn/SDE allocation
- middle SDE allocation
- late ODE allocation with a zero-slope tail in `t`-space

The structure module now supports:

- generalized `piecewise_power` schedules with asymmetric start/end powers
- optional endpoint trimming for the legacy piecewise schedule
- a new `phase_power` schedule generator

While porting the change onto `origin/main`, the existing upstream `rho: float = 0.7`
hotfix was preserved. No unrelated inference, alignment, or POC logic was moved.

## Validation

- `ruff format` on `src/kfold/model/modules/structure_module/kfold_ecsi.py`
- `ruff check` on `src/kfold/model/modules/structure_module/kfold_ecsi.py`

No tests were run.
