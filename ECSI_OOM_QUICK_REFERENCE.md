# ECSI CPU OOM - Quick Reference Guide

## 🔍 Investigation Summary

**Repository:** SeonghwanSeo/kfold  
**File Analyzed:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`  
**First Commit:** 4cd4f5e (retry CI/CD)  
**Total Issues Found:** 14  
**Analysis Date:** 2025-12-29

---

## 📊 Issues by Priority

| Priority | Count | Action Required |
|----------|-------|----------------|
| 🔴 CRITICAL | 1 | Immediate fix needed |
| 🟠 HIGH | 5 | Fix within sprint |
| 🟡 MEDIUM-HIGH | 1 | Monitor and plan fix |
| 🟢 MEDIUM | 7 | Optimize when possible |

---

## 🔴 Top 3 Critical Issues

### #1 - CRITICAL: Trajectory Accumulation ⚠️
- **Lines:** 647, 724
- **Method:** `sample_structure()`
- **Problem:** Stores all 200 intermediate states without limit
- **Memory Impact:** 7.68 GB in worst case
- **Fix:** Add trajectory downsampling or max length limit

### #2 - HIGH: Random Noise in Sampling Loop
- **Line:** 717
- **Method:** `sample_structure()`
- **Problem:** Creates new 38.4 MB tensor × 200 iterations
- **Memory Impact:** 7.68 GB cumulative
- **Fix:** Pre-allocate buffer, use in-place generation

### #3 - HIGH: Zeros Tensor in Loop
- **Line:** 664
- **Method:** `sample_structure()`
- **Problem:** Allocates 38.4 MB × 200 times unnecessarily
- **Memory Impact:** 7.68 GB cumulative
- **Fix:** Pre-allocate outside loop, reuse

---

## 🎯 Quick Fixes (Code Snippets)

### Fix #1: Limit Trajectory Storage
```python
# In Config class, add:
max_trajectory_samples: int = 50  # Store max 50 frames

# In sample_structure(), modify:
if return_traj:
    # Only store if we're within limit or sampling interval
    if len(traj) < self.max_trajectory_samples:
        traj.append(x_t.cpu())
    elif step_idx % (num_steps // self.max_trajectory_samples) == 0:
        traj.append(x_t.cpu())
```

### Fix #2: Pre-allocate Loop Tensors
```python
# Before loop, add:
x0_hat = torch.zeros_like(x_t)  # Allocate once
noise_buffer = torch.zeros_like(x_t)  # For random generation

# In loop, replace line 664:
# x0_hat = torch.zeros_like(x_t)  # DELETE THIS
# Already allocated before loop

# Replace line 717:
# noise = torch.randn_like(x_t)  # DELETE THIS
noise_buffer.normal_()  # In-place generation
noise = noise_buffer
```

### Fix #3: Add Configuration Validation
```python
# In Config class or __init__:
def validate_config(self):
    MAX_BATCH = 32
    MAX_SAMPLES = 50
    MAX_STEPS = 500
    
    assert self.num_steps <= MAX_STEPS, \
        f"num_steps {self.num_steps} exceeds maximum {MAX_STEPS}"
    
    # Add warning for large memory usage
    if self.num_steps > 200:
        import warnings
        warnings.warn(
            f"num_steps={self.num_steps} may cause high memory usage. "
            f"Estimated peak: {self._estimate_memory_gb():.2f} GB"
        )

def _estimate_memory_gb(self, batch_size=16, num_samples=20, num_atoms=10000):
    """Estimate peak memory in GB."""
    single_tensor_mb = batch_size * num_samples * num_atoms * 3 * 4 / (1024**2)
    trajectory_gb = single_tensor_mb * self.num_steps / 1024
    return trajectory_gb
```

---

## 📋 All Issues at a Glance

| # | Risk | Line(s) | Method | Issue | Memory Impact |
|---|------|---------|---------|-------|---------------|
| 1 | 🔴 CRITICAL | 647, 724 | `sample_structure()` | Trajectory storage | 7.68 GB |
| 2 | 🟠 HIGH | 522 | `interpolate()` | Random noise | 38.4 MB/call |
| 3 | 🟠 HIGH | 638 | `sample_structure()` | Prior sampling | 96 MB |
| 4 | 🟠 HIGH | 644 | `sample_structure()` | Tensor cloning | 2x memory |
| 5 | 🟠 HIGH | 664 | `sample_structure()` | Loop zeros | 7.68 GB |
| 6 | 🟠 HIGH | 717 | `sample_structure()` | Loop noise | 7.68 GB |
| 7 | 🟡 MED-HIGH | 382, 387 | `sample_noise_level()` | Unbounded sampling | Varies |
| 8 | 🟢 MEDIUM | 389 | `sample_noise_level()` | Beta sampling | Varies |
| 9 | 🟢 MEDIUM | 428 | `get_sampling_schedule()` | Schedule tensor | 40 KB |
| 10 | 🟢 MEDIUM | 519 | `interpolate()` | Intermediate | 38.4 MB |
| 11 | 🟢 MEDIUM | 690 | `sample_structure()` | z_hat compute | 38.4 MB |
| 12 | 🟢 MEDIUM | 710 | `sample_structure()` | Drift compute | 38.4 MB |
| 13 | 🟢 MEDIUM | 334 | `forward_model()` | Concatenation | 76.8 MB |
| 14 | 🟢 MEDIUM | 718 | `sample_structure()` | Diffusion scale | Small |

---

## 💾 Memory Scaling Table

| Config | Batch | Samples | Atoms | Steps | Traj? | Peak Memory |
|--------|-------|---------|-------|-------|-------|-------------|
| Minimal | 1 | 1 | 1000 | 50 | No | ~10 MB |
| Small | 4 | 5 | 2000 | 100 | No | ~340 MB |
| Medium | 8 | 10 | 5000 | 200 | No | ~2 GB |
| Large | 16 | 20 | 10000 | 200 | No | ~8 GB |
| **Danger** | 16 | 20 | 10000 | 200 | **Yes** | **~16 GB** ⚠️ |

**Formula:** 
```
Single Tensor = B × N × L × 3 × 4 bytes
Peak Memory ≈ Single Tensor × (1 + num_steps if return_traj else 5)
```

---

## 🚀 Recommended Actions

### Phase 1: Immediate (This Week)
- [ ] Implement trajectory length limit (#1)
- [ ] Add configuration validation
- [ ] Document memory requirements in README
- [ ] Add memory estimation utility

### Phase 2: Short-term (This Sprint)
- [ ] Pre-allocate loop tensors (#5, #6)
- [ ] Add memory profiling mode
- [ ] Implement warnings for large configs
- [ ] Add unit tests for memory limits

### Phase 3: Medium-term (Next Sprint)
- [ ] Optimize intermediate allocations (#10-14)
- [ ] Implement tensor reuse strategies
- [ ] Add chunking support for large batches
- [ ] Memory profiling in CI/CD

### Phase 4: Long-term (Future)
- [ ] Gradient checkpointing
- [ ] Mixed precision support
- [ ] On-disk trajectory storage
- [ ] Advanced memory optimization

---

## 📚 Related Documentation

- **Detailed Analysis:** `ECSI_CPU_OOM_ANALYSIS.md`
- **Annotated Source:** `ECSI_ANNOTATED_SOURCE.md`
- **Source Code:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`

---

## ⚙️ Testing Memory Issues

### Unit Test Example
```python
def test_memory_with_large_config():
    """Test that large configs either work or fail gracefully."""
    import tracemalloc
    
    tracemalloc.start()
    
    cfg = KFoldECSI.Config(
        num_steps=200,
        # ... other config
    )
    
    model = KFoldECSI(cfg, score_model)
    
    # Test with return_traj=False (should work)
    result = model.sample_structure(..., return_traj=False)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory: {peak / 1024**2:.2f} MB")
    
    assert peak < 5 * 1024**2 * 1024, "Memory usage too high"
    
    tracemalloc.stop()
```

### Stress Test
```bash
# Run with memory profiling
python -m memory_profiler scripts/inference.py --config large_config.yaml

# Monitor with psutil
python -c "
import psutil
import time
# Run your code here
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024**2:.2f} MB')
"
```

---

## 🔗 Git Information

**First ECSI Commit:**
```
commit 4cd4f5e
Author: [Author Name]
Date: [Date]

    retry CI/CD
    
    Files added:
    - src/kfold/model/modules/structure_module/kfold_ecsi.py (734 lines)
    - configs/model/module/structure_module/ecsi.yaml
    - tests/structure_module/test_ecsi_interpolation.py
    - tests/structure_module/test_ecsi_interpolation_synthetic.py
```

**To view the original code:**
```bash
git show 4cd4f5e:src/kfold/model/modules/structure_module/kfold_ecsi.py
```

---

## 📞 Contact

For questions about this analysis:
- See full report: `ECSI_CPU_OOM_ANALYSIS.md`
- Review annotated code: `ECSI_ANNOTATED_SOURCE.md`
- Check source: `src/kfold/model/modules/structure_module/kfold_ecsi.py`

---

**Last Updated:** 2025-12-29  
**Analysis Version:** 1.0  
**Status:** ✅ Complete
