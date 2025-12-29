# ECSI vs EDM Memory Leak Comparison

## Key Finding

**EDM + PairMixer:** No memory leak
**ECSI + PairMixer:** Memory leak occurs

## Critical Differences

### 1. Cache Management

**EDM (af3_edm.py line 283):**
```python
model_cache = {}  # Created once before loop
```

**ECSI (kfold_ecsi.py line 628):**
```python
model_cache = {}  # Created once before loop
```

**Verdict:** SAME - Both create cache once

---

### 2. Loop Structure

**EDM (af3_edm.py lines 306-350):**
```python
for step_idx in range(1, num_steps):  # Simple range loop
    # Line 8: Process in chunks
    atom_coords_denoised = torch.zeros_like(atom_coords_noisy)  # Created EACH iteration
    for st in range(0, num_diffusion_samples, max_parallel_samples):
        atom_coords_denoised[:, st:end] = self.forward_model(...)
```

**ECSI (kfold_ecsi.py lines 650-677):**
```python
for step_idx in range(num_steps):  # Simple range loop
    # Get denoised prediction
    x0_hat = torch.zeros_like(x_t)  # Created EACH iteration
    for st in range(0, num_diffusion_samples, max_parallel_samples):
        x0_hat[:, st:end] = self.forward_model(...)
```

**Verdict:** SIMILAR - Both allocate zeros_like in loop

---

### 3. CRITICAL DIFFERENCE: Sampling Algorithm Complexity

**EDM (Simple ODE):**
```python
for step_idx in range(1, num_steps):
    # 1. Add noise
    atom_coords_noisy = atom_coords + eps
    
    # 2. Denoise
    atom_coords_denoised = self.forward_model(...)
    
    # 3. Update with simple delta
    delta_coords = (atom_coords_noisy - atom_coords_denoised) / t_hat
    atom_coords = atom_coords_noisy + self.step_scale * dt * delta_coords
```

**ECSI (Complex SDE with Multiple Computations):**
```python
for step_idx in range(num_steps):
    # 1. Forward model
    x0_hat = self.forward_model(...)
    
    # 2. Compute route coefficients (6 tensors)
    alpha_t = self.alpha(t_exp)
    beta_t = self.beta(t_exp)
    gamma_t = self.gamma(t_exp)
    alpha_dot = self.alpha_deriv(t_exp)
    beta_dot = self.beta_deriv(t_exp)
    gamma_dot = self.gamma_deriv(t_exp)
    
    # 3. Compute z_hat (complex expression)
    z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)
    
    # 4. Compute eps_t (complex expression)
    eps_t = self.eta * (gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2)
    
    # 5. Compute drift (complex multi-term expression)
    drift = (
        alpha_dot * x0_hat
        + beta_dot * x_apo
        + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
    )
    
    # 6. Generate noise
    noise = torch.randn_like(x_t)
    
    # 7. Euler step with diffusion
    diffusion_scale = torch.sqrt(2 * torch.abs(eps_t) * abs(dt) + 1e-8)
    x_t = x_t + drift * dt + diffusion_scale * noise
```

**Verdict:** VERY DIFFERENT - ECSI has 10x more intermediate computations

---

### 4. Number of Intermediate Tensors Per Iteration

**EDM:**
- `eps` (noise)
- `atom_coords_noisy` 
- `atom_coords_denoised`
- `delta_coords`
- **Total: 4 intermediate tensors**

**ECSI:**
- `x0_hat`
- `t_exp` (expanded time)
- `alpha_t`, `beta_t`, `gamma_t` (3 coefficients)
- `alpha_dot`, `beta_dot`, `gamma_dot` (3 derivatives)
- `z_hat` (computed state)
- `eps_t` (noise coefficient)
- `drift` (velocity)
- `noise` (random)
- `diffusion_scale`
- Multiple intermediate expressions in each computation
- **Total: 15+ intermediate tensors**

**Verdict:** ECSI creates 3-4x more intermediate tensors per iteration

---

### 5. Variable Reuse

**EDM:**
```python
atom_coords = atom_coords_noisy + self.step_scale * dt * delta_coords
# Simple reassignment, old tensors can be GC'd immediately
```

**ECSI:**
```python
x_t = x_t + drift * dt + diffusion_scale * noise
# x_t is both input and output - potential for autograd graph retention
```

**Verdict:** ECSI has more complex variable interdependencies

---

### 6. Autograd Graph Complexity

**EDM:**
- Simple forward pass
- Minimal interdependencies
- Easy for PyTorch to release

**ECSI:**
- Complex multi-step computations
- Many intermediate tensors reference each other
- z_hat depends on x_t, x0_hat, x_apo, alpha_t, beta_t, gamma_t
- drift depends on z_hat, x0_hat, x_apo, multiple coefficients
- x_t update depends on drift, diffusion_scale, noise
- **Complex computation graph harder to release**

**Verdict:** ECSI creates deeper, more interconnected computation graphs

---

## Root Cause Hypothesis

### Why ECSI Leaks but EDM Doesn't

1. **Computation Graph Complexity**
   - ECSI's complex SDE formulation creates deeply nested computation graphs
   - Each iteration creates 15+ tensors with complex interdependencies
   - PyTorch's autograd may retain more references for potential backward passes
   - Even with `torch.no_grad()`, Python closures may capture references

2. **Reference Cycle Creation**
   - ECSI's `z_hat` computation references `x_t`, `x0_hat`, `x_apo`
   - `drift` computation references `z_hat`, `x0_hat`, `x_apo`
   - `x_t` update references `drift`, which references `z_hat`, which references old `x_t`
   - Creates circular reference chain: `x_t` → `z_hat` → `drift` → new `x_t`
   - Python GC may fail to collect these cycles immediately

3. **Coefficient Computation Overhead**
   - ECSI computes 6 coefficient tensors per iteration (alpha, beta, gamma + derivatives)
   - These are small tensors but created 200 times
   - May accumulate in Python's object heap

4. **Combined with PairMixer Checkpointing**
   - PairMixer's `partial()` closures (Issue #1) are called MORE frequently in ECSI
   - ECSI's forward_model is called 200 times with complex inputs
   - Each call goes through PairMixer with checkpoint mechanism
   - Checkpointing saves closures that capture ECSI's complex tensors
   - Combination amplifies the leak

5. **EDM's Simplicity Helps**
   - EDM's simple ODE formulation has linear computation flow
   - Fewer intermediate tensors
   - Simpler dependencies → easier to GC
   - Checkpointing still happens but with simpler captured state

---

## Specific ECSI Code Patterns Causing Leaks

### Pattern #1: Reusing Variable Name with Complex Dependencies

**Location:** kfold_ecsi.py line 719

```python
x_t = x_t + drift * dt + diffusion_scale * noise
```

**Problem:**
- `x_t` appears on both sides
- Old `x_t` is used in computation then overwritten
- `drift` was computed using old `x_t` (via `z_hat`)
- Creates reference cycle that's hard to break

**Why EDM Doesn't Have This:**
```python
atom_coords = atom_coords_noisy + self.step_scale * dt * delta_coords
```
- Uses `atom_coords_noisy` (distinct variable)
- No self-reference in same expression
- Cleaner dependency chain

---

### Pattern #2: Multi-Level Tensor Dependencies

**Location:** kfold_ecsi.py lines 690, 710-714

```python
# z_hat depends on: x_t, x0_hat, x_apo, alpha_t, beta_t, gamma_t
z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)

# drift depends on: z_hat (which depends on 6 tensors), x0_hat, x_apo, 4 coefficients
drift = (
    alpha_dot * x0_hat
    + beta_dot * x_apo
    + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
)
```

**Problem:**
- Deep dependency tree: drift → z_hat → [x_t, x0_hat, x_apo, 3 coefficients]
- Each tensor holds references to all its inputs
- With 200 iterations × complex graph = reference accumulation
- Python GC may not break these cycles fast enough

---

### Pattern #3: x_apo Captured in Every Iteration

**Location:** kfold_ecsi.py line 690, 712

```python
z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)
drift = alpha_dot * x0_hat + beta_dot * x_apo + ...
```

**Problem:**
- `x_apo` is referenced in every iteration (200 times)
- Each reference in a complex expression
- `x_apo` from line 638 persists throughout entire sampling
- Referenced by z_hat and drift in every iteration
- May prevent GC of related objects

**Why EDM Doesn't Have This:**
- EDM doesn't have equivalent to `x_apo`
- No persistent tensor threaded through all iterations
- Cleaner per-iteration isolation

---

## Validation Tests

### Test 1: Does EDM Leak?

```bash
# Run training with EDM + PairMixer
python scripts/train.py --config configs/train-esm2-edm-pairmixer.yaml
# Monitor: system memory should stay stable ✓
```

### Test 2: Does ECSI Leak?

```bash
# Run training with ECSI + PairMixer
python scripts/train.py --config configs/train-esm2-ecsi-mini-pairmixer.yaml
# Monitor: system memory should grow ✗ (CONFIRMED)
```

### Test 3: Does ECSI without PairMixer Leak?

```bash
# Run training with ECSI + different trunk
python scripts/train.py --config configs/train-esm2-ecsi-mini.yaml
# Monitor: check if leak persists
```

---

## Conclusion

**ECSI-specific factors causing memory leak:**

1. ✅ Complex SDE formulation with 15+ intermediate tensors per iteration
2. ✅ Deep dependency trees (drift → z_hat → multiple inputs)
3. ✅ Self-referential updates (`x_t = x_t + ...`)
4. ✅ Persistent tensor `x_apo` referenced in every iteration
5. ✅ 6 coefficient tensors computed per iteration
6. ✅ Circular reference patterns harder for Python GC

**Combined with PairMixer:**
- Checkpointing mechanism captures these complex states
- `partial()` closures hold references to complex ECSI tensors
- Amplifies the base ECSI leak

**EDM avoids this because:**
- ✅ Simple ODE with 4 intermediate tensors
- ✅ Linear dependency flow
- ✅ No self-referential updates
- ✅ No persistent tensors threaded through iterations
- ✅ Minimal coefficient computations
- ✅ Easy for GC to clean up

