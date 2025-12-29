# ECSI CPU OOM Investigation - Documentation Index

## 📋 Overview

This directory contains a comprehensive analysis of CPU Out-of-Memory (OOM) risks in the ECSI (Endpoint-Conditioned Stochastic Interpolant) implementation in the kfold repository.

**Investigation Date:** 2025-12-29  
**Analyzed File:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`  
**First ECSI Commit:** `4cd4f5e` (retry CI/CD)  
**Total Issues Identified:** 14

---

## 📚 Documentation Files

### 1. Quick Reference Guide (Start Here) ⭐
**File:** [`ECSI_OOM_QUICK_REFERENCE.md`](ECSI_OOM_QUICK_REFERENCE.md) (7.6 KB, 264 lines)

**Best for:** Developers who need quick fixes and actionable items

**Contents:**
- Top 3 critical issues summary
- Quick fix code snippets
- Memory scaling table
- Phased action plan
- Testing guidance
- All issues at a glance table

---

### 2. Comprehensive Analysis Report
**File:** [`ECSI_CPU_OOM_ANALYSIS.md`](ECSI_CPU_OOM_ANALYSIS.md) (16 KB, 576 lines)

**Best for:** Deep dive into each issue with full context

**Contents:**
- Detailed description of all 14 issues
- Risk level classification (CRITICAL/HIGH/MEDIUM)
- Memory impact calculations for each issue
- Worst-case scenario analysis (up to 8 GB)
- Realistic scenario analysis
- 10 comprehensive recommendations
- Testing recommendations
- Memory scaling formulas

**Key Sections:**
- Issue #1-14 detailed breakdowns
- Memory Scaling Analysis
- Recommendations Summary (Immediate/Medium/Long-term)
- Testing Recommendations
- Conclusion and Priority Order

---

### 3. Annotated Source Code
**File:** [`ECSI_ANNOTATED_SOURCE.md`](ECSI_ANNOTATED_SOURCE.md) (13 KB, 366 lines)

**Best for:** Code reviewers and developers implementing fixes

**Contents:**
- Key source code sections with inline annotations
- Visual risk indicators (🔴🟠🟡🟢)
- Memory impact for each code location
- Context-specific recommendations
- Location-by-location analysis

**Legend:**
- 🔴 CRITICAL - Immediate action required
- 🟠 HIGH - Should be addressed soon
- 🟡 MEDIUM-HIGH - Notable concern
- 🟢 MEDIUM - Monitor and optimize

---

## 🎯 Key Findings Summary

### Critical Statistics

| Metric | Value |
|--------|-------|
| **Total Issues** | 14 |
| **Critical Issues** | 1 |
| **High-Risk Issues** | 5 |
| **Worst-Case Memory** | ~8 GB |
| **Primary Culprit** | Trajectory storage (7.68 GB) |
| **Most Problematic Method** | `sample_structure()` (8 issues) |

### Top 3 Issues to Fix

1. **🔴 CRITICAL - Trajectory Accumulation**
   - Lines: 647, 724
   - Impact: 7.68 GB with default settings
   - Fix: Add trajectory downsampling/limits

2. **🟠 HIGH - Random Noise in Loop**
   - Line: 717
   - Impact: 7.68 GB cumulative
   - Fix: Pre-allocate buffer, use in-place generation

3. **🟠 HIGH - Zeros Tensor in Loop**
   - Line: 664
   - Impact: 7.68 GB cumulative
   - Fix: Pre-allocate outside loop

### Memory Scaling

| Configuration | Memory Usage |
|---------------|--------------|
| Minimal (B=1, N=1, L=1k) | ~10 MB ✅ |
| Small (B=4, N=5, L=2k) | ~340 MB ✅ |
| Medium (B=8, N=10, L=5k) | ~2 GB ⚠️ |
| Large (B=16, N=20, L=10k) | ~8 GB ❌ |
| **With Trajectory** | **~16 GB** 🔴 |

---

## 🚀 Recommended Reading Order

### For Quick Fixes
1. Read [`ECSI_OOM_QUICK_REFERENCE.md`](ECSI_OOM_QUICK_REFERENCE.md)
2. Implement the top 3 fixes from the "Quick Fixes" section
3. Add configuration validation

### For Comprehensive Understanding
1. Start with [`ECSI_OOM_QUICK_REFERENCE.md`](ECSI_OOM_QUICK_REFERENCE.md) for overview
2. Read [`ECSI_CPU_OOM_ANALYSIS.md`](ECSI_CPU_OOM_ANALYSIS.md) for detailed analysis
3. Review [`ECSI_ANNOTATED_SOURCE.md`](ECSI_ANNOTATED_SOURCE.md) while coding

### For Code Review
1. Keep [`ECSI_ANNOTATED_SOURCE.md`](ECSI_ANNOTATED_SOURCE.md) open
2. Cross-reference with actual source code
3. Use the "All Issues at a Glance" table in [`ECSI_OOM_QUICK_REFERENCE.md`](ECSI_OOM_QUICK_REFERENCE.md)

---

## 🔧 Quick Fixes (Copy-Paste Ready)

### Fix #1: Limit Trajectory Storage (Most Critical)
```python
# In KFoldECSI.Config class:
max_trajectory_samples: int = 50  # Limit trajectory storage

# In sample_structure() method, replace lines 647 and 724:
if return_traj:
    if len(traj) < self.max_trajectory_samples:
        traj.append(x_t.cpu())
    elif len(traj) < self.max_trajectory_samples and \
         step_idx % (num_steps // self.max_trajectory_samples) == 0:
        traj.append(x_t.cpu())
```

### Fix #2: Pre-allocate Loop Tensors
```python
# In sample_structure(), before the main loop (after line 649):
x0_hat = torch.zeros_like(x_t)  # Pre-allocate
noise_buffer = torch.zeros_like(x_t)  # Pre-allocate

# In the loop:
# DELETE line 664: x0_hat = torch.zeros_like(x_t)
# REPLACE line 717: noise = torch.randn_like(x_t)
# WITH:
noise_buffer.normal_()
noise = noise_buffer
```

### Fix #3: Add Configuration Validation
```python
# In KFoldECSI.__init__() method:
def __init__(self, cfg: Config, score_model: BaseScoreModel):
    super().__init__(cfg, score_model)
    # ... existing code ...
    
    # Add validation
    self._validate_memory_config()

def _validate_memory_config(self):
    """Validate configuration to prevent OOM."""
    if self.num_steps > 500:
        raise ValueError(
            f"num_steps={self.num_steps} exceeds maximum 500. "
            f"This may cause OOM errors."
        )
    
    if self.num_steps > 200:
        import warnings
        warnings.warn(
            f"num_steps={self.num_steps} may use significant memory. "
            f"Consider reducing for large structures."
        )
```

---

## 📊 Issue Distribution

### By Risk Level
```
CRITICAL:     █ 1 issue  (7%)
HIGH:         █████ 5 issues (36%)
MEDIUM-HIGH:  █ 1 issue  (7%)
MEDIUM:       ███████ 7 issues (50%)
```

### By Method
```
sample_structure():     ████████ 8 issues (57%)
interpolate():          ██ 2 issues (14%)
sample_noise_level():   ███ 3 issues (21%)
forward_model():        █ 1 issue (7%)
```

### By Type
```
Loop allocations:       ████ 4 issues (29%)
Unbounded operations:   ███ 3 issues (21%)
Intermediate tensors:   ████ 4 issues (29%)
Configuration issues:   ███ 3 issues (21%)
```

---

## 🧪 Testing Your Fixes

After implementing fixes, test with:

```bash
# Run existing tests
pytest tests/structure_module/test_ecsi_interpolation.py
pytest tests/structure_module/test_ecsi_interpolation_synthetic.py

# Add memory profiling
python -m memory_profiler scripts/inference.py --config ecsi_config.yaml

# Monitor memory usage
python -c "
import torch
import psutil
import gc

# Your test code here
process = psutil.Process()
print(f'Memory before: {process.memory_info().rss / 1024**2:.2f} MB')

# Run ECSI code
# ...

gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f'Memory after: {process.memory_info().rss / 1024**2:.2f} MB')
"
```

---

## 📈 Implementation Phases

### Phase 1: Immediate (This Week) ⚡
- [ ] Implement trajectory length limit
- [ ] Add configuration validation
- [ ] Document memory requirements
- [ ] Test with large configurations

### Phase 2: Short-term (This Sprint) 🎯
- [ ] Pre-allocate loop tensors
- [ ] Add memory profiling
- [ ] Implement warnings
- [ ] Add memory unit tests

### Phase 3: Medium-term (Next Sprint) 🔄
- [ ] Optimize intermediate allocations
- [ ] Implement tensor reuse
- [ ] Add chunking support
- [ ] Memory profiling in CI/CD

### Phase 4: Long-term (Roadmap) 🚀
- [ ] Gradient checkpointing
- [ ] Mixed precision support
- [ ] On-disk trajectory storage
- [ ] Advanced optimizations

---

## 📞 Additional Resources

### Related Files
- **Source Code:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`
- **Config:** `configs/model/module/structure_module/ecsi.yaml`
- **Tests:** `tests/structure_module/test_ecsi_*.py`

### Git Information
```bash
# View original commit
git show 4cd4f5e

# View ECSI file history
git log --follow -- src/kfold/model/modules/structure_module/kfold_ecsi.py

# View specific version
git show 4cd4f5e:src/kfold/model/modules/structure_module/kfold_ecsi.py
```

### Memory Estimation Formula
```python
def estimate_memory_mb(batch_size, num_samples, num_atoms, num_steps, return_traj=False):
    """Estimate peak memory usage in MB."""
    single_tensor = batch_size * num_samples * num_atoms * 3 * 4 / (1024**2)
    if return_traj:
        return single_tensor * num_steps  # Trajectory dominates
    else:
        return single_tensor * 5  # Various intermediates
```

---

## ✅ Checklist for Reviewers

Before merging ECSI-related changes:

- [ ] Check for new loop allocations (Issues #5, #6)
- [ ] Verify trajectory storage has limits (Issue #1)
- [ ] Ensure large tensors are validated (Issues #2, #3)
- [ ] Look for unnecessary `.clone()` operations (Issue #4)
- [ ] Confirm configuration has reasonable defaults
- [ ] Test with large structures (>5000 atoms)
- [ ] Profile memory usage
- [ ] Document memory requirements

---

## 🙏 Acknowledgments

This analysis was created to help identify and prevent CPU OOM issues in the kfold ECSI implementation. The analysis is based on:

- Static code analysis of `kfold_ecsi.py` (734 lines)
- Memory scaling calculations
- PyTorch memory management patterns
- Best practices for large tensor operations

---

## 📝 Version History

- **v1.0** (2025-12-29): Initial comprehensive analysis
  - 14 issues identified
  - 3 documentation files created
  - Memory scaling analysis completed
  - Recommendations provided

---

## 🔗 Quick Links

| Document | Purpose | Size |
|----------|---------|------|
| [Quick Reference](ECSI_OOM_QUICK_REFERENCE.md) | Fast lookup & fixes | 7.6 KB |
| [Full Analysis](ECSI_CPU_OOM_ANALYSIS.md) | Detailed investigation | 16 KB |
| [Annotated Source](ECSI_ANNOTATED_SOURCE.md) | Code-level review | 13 KB |

---

**Last Updated:** 2025-12-29  
**Status:** ✅ Analysis Complete  
**Next Steps:** Implement recommended fixes

*For questions or updates, refer to the detailed documentation files listed above.*
