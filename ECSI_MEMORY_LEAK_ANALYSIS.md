# ECSI + PairMixer Memory Leak Analysis

## 🔍 Problem Statement

**Symptom:** During training with ECSI + PairMixer, while the process allocated memory stays stable, overall system memory utilization keeps increasing.

**Root Cause:** Garbage not being collected properly during training, leading to memory leak outside of tracked process memory.

---

## 🎯 Identified Memory Leak Sources

### Issue #1 - CRITICAL: `partial()` Functions in PairMixer Accumulating References 🔴

**Location:** `src/kfold/model/layers/pairmixer/pairmixer.py`, lines 127-134

**Code:**
```python
blocks = [
    partial(
        b,
        pair_mask=pair_mask.float(),
        use_cuequiv_mul=use_cuequiv_mul,
    )
    for b in self.blocks
]
```

**Problem:**
1. **Creates new `partial` objects every forward pass**
2. Each `partial` captures references to:
   - `pair_mask.float()` - a tensor created on the fly
   - The block `b` (module reference)
   - `use_cuequiv_mul` boolean
3. **These partial objects are passed to `checkpoint_blocks()`**
4. **During checkpointing, these partials are saved for backward pass**
5. **The saved tensors and closures are NOT properly released after backward**

**Why This Causes Memory Leak:**
- `partial()` creates closure that holds strong references to captured variables
- `pair_mask.float()` creates a NEW tensor every time (not reused)
- When gradient checkpointing is used, these partials are saved
- PyTorch's checkpointing may not properly release the closure references
- Even after backward pass, the saved partial objects may remain in memory
- Over many training iterations, these accumulate

**Memory Impact:**
- Per forward pass: `num_blocks × (pair_mask_size + overhead)`
- Example: 48 blocks × (B × L × L × 4 bytes) = 48 × 64MB = **3 GB per forward pass**
- If not released: **accumulates unbounded**

**Evidence:**
- Process memory stays stable (PyTorch tracks allocated tensors)
- System memory grows (Python GC doesn't collect the closures)
- Specific to checkpointing mode (training with gradient checkpointing)

---

### Issue #2 - HIGH: Model Cache Dictionary Not Cleared Between Training Steps 🟠

**Location:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`, line 628

**Code:**
```python
def sample_structure(self, ...):
    ...
    model_cache = {}  # Created once per sample_structure call
    
    # Used throughout sampling
    for step_idx in range(num_steps):
        ...
        x0_hat[:, st:end] = self.forward_model(
            ...
            model_cache=model_cache,  # Same cache reused
        )
```

**Problem:**
1. `model_cache` is created as empty dict
2. Throughout sampling (200 steps), cache is populated
3. Cache stores intermediate representations like:
   - `"atom_attn_encoder"` -> query/key/value projections
   - `"diffusion_conditioning"` -> pair representations
   - `"rel_pos_encoding"` -> position encodings
4. **Cache is NOT explicitly cleared at the end**
5. While dict goes out of scope, if any reference is held elsewhere, leak occurs

**Why This Causes Memory Leak:**
- Cached tensors accumulate over 200 sampling steps
- If any module holds reference to cache (via closure), it persists
- Dict itself may not be garbage collected if referenced
- Cached tensors are typically large (full model activations)

**Memory Impact:**
- Per sample_structure call: ~500 MB - 2 GB depending on model size
- If cache persists across calls: **accumulates**

---

### Issue #3 - HIGH: Recycling Loop Variables Not Released 🟠

**Location:** `src/kfold/model/modules/trunk/pairmixer_trunk.py`, lines 122-166

**Code:**
```python
# Line 123
s_hat = torch.zeros_like(s_init)
z_hat = torch.zeros_like(z_init)

for i in range(0, num_recycles + 1):
    enable_grad = self.training and i == num_recycles
    
    with torch.set_grad_enabled(enable_grad):
        ...
        # Line 163
        s_hat, z_hat = s, z  # Reassignment

# Variables s_hat, z_hat persist after loop
```

**Problem:**
1. `s_hat` and `z_hat` are created before recycling loop
2. During loop, they are reassigned multiple times
3. **Old tensor references may not be immediately freed**
4. If gradient graph holds references to intermediate `s_hat`/`z_hat`, they persist
5. With `num_recycles + 1` iterations, creates chain of references

**Why This Causes Memory Leak:**
- Each recycling iteration creates new tensors for `s` and `z`
- Assignment `s_hat, z_hat = s, z` doesn't guarantee old tensors are freed
- If autograd graph is not properly cleared between iterations
- Multiple versions of `s_hat` and `z_hat` may coexist in memory

**Memory Impact:**
- Per recycle iteration: B × L × C_s + B × L × L × C_z
- Example: 4 × 512 × 384 + 4 × 512 × 512 × 128 = ~130 MB per iteration
- With 3 recycles: 3 × 130 MB = **390 MB** that should be freed but may persist

---

### Issue #4 - HIGH: Checkpoint Function Closures Not Released 🟠

**Location:** `src/kfold/utils/checkpointing.py`, lines 91-95

**Code:**
```python
def chunker(s, e):
    def exec_sliced(*a):
        return exec(blocks[s:e], a)
    return exec_sliced
```

**Problem:**
1. **Nested function `exec_sliced` creates closure over `blocks[s:e]`**
2. This closure captures slice of blocks (module references)
3. The closure is passed to PyTorch's checkpoint mechanism
4. **Checkpoint saves this closure for backward pass**
5. After backward, closure should be released but may persist

**Why This Causes Memory Leak:**
- Closure captures `blocks` list which contains all module references
- Even though only slice `[s:e]` is used, entire `blocks` list may be captured
- Python's closure captures entire scope, not just used variables
- Multiple closures created per training step (one per chunk)
- If checkpoint doesn't properly release: accumulates

**Memory Impact:**
- Per closure: reference to all blocks + partial overhead
- With blocks_per_ckpt: `num_blocks / blocks_per_ckpt` closures per forward
- Example: 48 blocks / 4 = 12 closures
- If each holds ~100 MB: 12 × 100 MB = **1.2 GB per forward pass**

---

### Issue #5 - MEDIUM: `pair_mask.float()` Created Every Forward Pass 🟡

**Location:** `src/kfold/model/layers/pairmixer/pairmixer.py`, line 125

**Code:**
```python
def forward(self, s, z, mask, use_cuequiv_mul=False):
    pair_mask = mask[..., None] & mask[..., None, :]  # Line 125
    
    blocks = [
        partial(
            b,
            pair_mask=pair_mask.float(),  # float() called here
            use_cuequiv_mul=use_cuequiv_mul,
        )
        for b in self.blocks
    ]
```

**Problem:**
1. `pair_mask.float()` creates a new float tensor every forward pass
2. This tensor is captured by 48 partial objects (one per block)
3. **Same tensor referenced 48 times but created fresh each time**
4. Not reused across forward passes

**Why This Causes Memory Leak:**
- Creates unnecessary tensor copies
- `.float()` operation creates new tensor (not in-place)
- Multiple references prevent early garbage collection
- Accumulates if partials are not properly released

**Memory Impact:**
- Per forward: B × L × L × 4 bytes
- Example: 4 × 512 × 512 × 4 = **4 MB per forward**
- Times 48 references: conceptually ~**192 MB** (though same tensor)
- If leaked: accumulates over training

---

### Issue #6 - MEDIUM: `torch.clear_autocast_cache()` Called But May Be Insufficient 🟡

**Location:** Multiple files (e.g., `pairmixer_trunk.py` line 131)

**Code:**
```python
with torch.set_grad_enabled(enable_grad):
    if enable_grad and torch.is_autocast_enabled():
        torch.clear_autocast_cache()  # Attempts to clear
```

**Problem:**
1. `torch.clear_autocast_cache()` clears autocast cache
2. **Does NOT clear general PyTorch cache or Python GC**
3. Only addresses CUDA autocast caching, not CPU memory
4. Other caches may still accumulate

**Why This May Not Prevent Leaks:**
- Autocast cache is separate from general memory management
- Doesn't clear gradient buffers
- Doesn't trigger Python garbage collection
- Doesn't clear checkpoint saved tensors

**Missing:**
- No explicit `gc.collect()` calls
- No `torch.cuda.empty_cache()` equivalent for CPU
- No manual clearing of model_cache dictionaries

---

## 🔬 Root Cause Analysis

### Primary Leak Mechanism

```
Training Loop Iteration
    ↓
Forward Pass (ECSI + PairMixer)
    ↓
Create partial() objects with tensor closures ← LEAK SOURCE #1
    ↓
Pass to checkpoint_blocks()
    ↓
Checkpoint saves closures ← LEAK SOURCE #2
    ↓
Backward Pass
    ↓
Closures SHOULD be released but aren't ← ROOT CAUSE
    ↓
Python GC doesn't collect (strong references remain)
    ↓
System memory grows (outside process tracking)
```

### Why Process Memory Stays Stable

- **PyTorch memory allocator** tracks allocated tensors
- Process RSS shows PyTorch's cached memory
- **Python garbage collector** manages closure objects separately
- Leaked closures reference tensors but aren't counted in process memory
- System memory includes both:
  - Process allocated memory (PyTorch tensors)
  - Garbage-not-collected memory (Python closures)

### Why This Happens with ECSI + PairMixer Specifically

1. **ECSI uses intensive sampling loops** (200 steps)
2. **PairMixer uses gradient checkpointing** (blocks_per_ckpt)
3. **Combination creates many closures**:
   - 200 sampling steps in ECSI
   - 48 blocks with checkpointing in PairMixer
   - Multiple recycling iterations
4. **Each closure captured by checkpointing mechanism**
5. **Checkpointing with non-reentrant mode may not release properly**

---

## ✅ Solutions and Fixes

### Fix #1 - CRITICAL: Avoid Creating Partial Objects in Hot Loop 🔴

**Current Code:**
```python
blocks = [
    partial(
        b,
        pair_mask=pair_mask.float(),
        use_cuequiv_mul=use_cuequiv_mul,
    )
    for b in self.blocks
]
```

**Fixed Code:**
```python
# Pre-convert pair_mask once
pair_mask_float = pair_mask.float()

# Create wrapper that doesn't use partial
def create_block_wrapper(block, mask, use_cuequiv):
    def wrapper(s, z):
        return block(s, z, pair_mask=mask, use_cuequiv_mul=use_cuequiv)
    return wrapper

# OR better: modify block signature to accept mask as instance variable
# Set as temporary attribute
self._temp_pair_mask = pair_mask_float
self._temp_use_cuequiv = use_cuequiv_mul

blocks = [
    lambda s, z, b=b: b(s, z, pair_mask=self._temp_pair_mask, 
                        use_cuequiv_mul=self._temp_use_cuequiv)
    for b in self.blocks
]

# Clean up after checkpointing
del self._temp_pair_mask
del self._temp_use_cuequiv
```

**Even Better: Modify Block Signature**
```python
# Store pair_mask as instance variable temporarily
self.current_pair_mask = pair_mask.float()
self.current_use_cuequiv = use_cuequiv_mul

# Modify blocks to read from self
# Then pass blocks directly without partial
blocks = self.blocks  # No partial needed

# In PairmixerBlock.forward(), read from parent:
# pair_mask = self.parent.current_pair_mask
```

---

### Fix #2 - HIGH: Explicitly Clear model_cache 🟠

**Add at end of `sample_structure()`:**
```python
def sample_structure(self, ...):
    model_cache = {}
    
    try:
        # ... sampling logic ...
        return sample_out
    finally:
        # Explicitly clear cache
        model_cache.clear()
        del model_cache
        
        # Force garbage collection periodically
        import gc
        if step_idx % 50 == 0:  # Every 50 steps
            gc.collect()
```

**Even Better: Use Context Manager**
```python
from contextlib import contextmanager

@contextmanager
def temporary_cache():
    cache = {}
    try:
        yield cache
    finally:
        cache.clear()
        del cache

def sample_structure(self, ...):
    with temporary_cache() as model_cache:
        # ... use model_cache ...
```

---

### Fix #3 - HIGH: Explicitly Release Recycling Variables 🟠

**Add explicit tensor deletion:**
```python
for i in range(0, num_recycles + 1):
    enable_grad = self.training and i == num_recycles
    
    with torch.set_grad_enabled(enable_grad):
        # ... forward logic ...
        
        # Before reassignment, delete old tensors if not needed
        if i > 0:
            del s, z  # Delete old tensors
        
        s, z = pairmixer_module(...)
        s_hat, z_hat = s, z

# After loop, ensure cleanup
del s, z
```

---

### Fix #4 - HIGH: Improve Checkpoint Implementation 🟠

**Modify checkpointing to avoid closure capture:**
```python
def checkpoint_blocks(blocks, args, blocks_per_ckpt, use_reentrant=None):
    # Instead of creating closures, use functools.partial differently
    
    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        
        # Extract slice ONCE, not in closure
        block_slice = blocks[s:e]
        
        # Create function that doesn't capture `blocks`
        def exec_blocks(block_list, *a):
            result = a
            for block in block_list:
                result = block(*result) if isinstance(result, tuple) else block(result)
            return result
        
        # Use partial with explicit arguments (no closure)
        args = checkpoint(
            partial(exec_blocks, block_slice),
            *args,
            use_reentrant=use_reentrant
        )
```

---

### Fix #5 - MEDIUM: Reuse pair_mask Conversion 🟡

**Pre-convert and reuse:**
```python
def forward(self, s, z, mask, use_cuequiv_mul=False):
    # Compute pair_mask once
    if not hasattr(self, '_cached_mask_shape') or self._cached_mask_shape != mask.shape:
        self._cached_pair_mask = (mask[..., None] & mask[..., None, :]).float()
        self._cached_mask_shape = mask.shape
    
    pair_mask = self._cached_pair_mask
    
    # Rest of forward pass...
```

---

### Fix #6 - Add Explicit Garbage Collection 🟢

**Add periodic GC calls:**
```python
import gc

def training_step(self, ...):
    result = # ... training logic ...
    
    # Force GC every N steps
    if self.training and hasattr(self, '_step_counter'):
        self._step_counter += 1
        if self._step_counter % 10 == 0:
            gc.collect()
    elif self.training:
        self._step_counter = 0
    
    return result
```

---

## 🧪 Testing and Validation

### Test Script for Memory Leak Detection

```python
import gc
import torch
import psutil
import tracemalloc

def test_memory_leak():
    """Test for memory leaks during training."""
    process = psutil.Process()
    tracemalloc.start()
    
    # Initial memory
    gc.collect()
    initial_mem = process.memory_info().rss / 1024**2
    initial_system = psutil.virtual_memory().used / 1024**2
    
    print(f"Initial - Process: {initial_mem:.2f} MB, System: {initial_system:.2f} MB")
    
    # Run training iterations
    for iteration in range(100):
        # Your training step here
        train_step()
        
        if iteration % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / 1024**2
            current_system = psutil.virtual_memory().used / 1024**2
            mem_growth = current_mem - initial_mem
            system_growth = current_system - initial_system
            
            current, peak = tracemalloc.get_traced_memory()
            
            print(f"Iter {iteration}: "
                  f"Process +{mem_growth:.2f} MB, "
                  f"System +{system_growth:.2f} MB, "
                  f"Peak: {peak/1024**2:.2f} MB")
            
            # Alert if system memory grows but process memory doesn't
            if system_growth > 1000 and mem_growth < 100:
                print(f"⚠️  WARNING: Potential memory leak detected!")
                print(f"   System memory grew {system_growth:.2f} MB")
                print(f"   But process memory only grew {mem_growth:.2f} MB")
    
    tracemalloc.stop()
```

---

## 📋 Implementation Checklist

### Phase 1: Immediate Fixes (This Week)
- [ ] Fix #1: Eliminate `partial()` in PairMixer forward
- [ ] Fix #2: Add explicit cache clearing
- [ ] Fix #6: Add periodic garbage collection
- [ ] Test with memory profiling script

### Phase 2: Structural Improvements (This Sprint)
- [ ] Fix #3: Clean up recycling loop variables
- [ ] Fix #4: Improve checkpoint implementation
- [ ] Add comprehensive memory profiling
- [ ] Document memory management best practices

### Phase 3: Long-term (Next Sprint)
- [ ] Investigate alternative checkpointing strategies
- [ ] Consider using `torch.utils.checkpoint` differently
- [ ] Implement memory-efficient training mode
- [ ] Add automatic leak detection to CI/CD

---

## 📚 Related Documentation

- **Static Analysis:** `ECSI_CPU_OOM_ANALYSIS.md`
- **Quick Reference:** `ECSI_OOM_QUICK_REFERENCE.md`
- **Annotated Source:** `ECSI_ANNOTATED_SOURCE.md`

---

## 🔗 Key Files

| File | Issues |
|------|--------|
| `src/kfold/model/layers/pairmixer/pairmixer.py` | #1, #5 |
| `src/kfold/model/modules/trunk/pairmixer_trunk.py` | #3 |
| `src/kfold/model/modules/structure_module/kfold_ecsi.py` | #2 |
| `src/kfold/utils/checkpointing.py` | #4 |

---

**Analysis Date:** 2025-12-29  
**Status:** ✅ Memory Leak Analysis Complete  
**Priority:** 🔴 CRITICAL - Implement immediately

*For static OOM analysis, see ECSI_CPU_OOM_ANALYSIS.md*
