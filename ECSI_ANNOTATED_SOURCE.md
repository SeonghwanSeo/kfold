# Annotated ECSI Source Code - CPU OOM Risk Markers

This document shows key sections of the ECSI source code with inline annotations marking potential CPU OOM issues.

## Legend
- 🔴 **CRITICAL** - Immediate action required
- 🟠 **HIGH** - Should be addressed soon
- 🟡 **MEDIUM-HIGH** - Notable concern
- 🟢 **MEDIUM** - Monitor and optimize

---

## Issue #13 🟢 MEDIUM: Tensor Concatenation (Line 334)

**Location:** `forward_model()` method

```python
        if self.normalize_data_end and not self.normalize_coordinate:
            prior_coords = prior_coords / self.sigma_data_end
        # 🟢 MEDIUM RISK: Tensor concatenation doubles last dimension (3→6)
        # Memory: Creates new tensor of size (B, N, La, 6) = ~76.8 MB for typical configs
        # Impact: Doubles memory for this variable
        r_noisy = torch.cat([r_noisy, prior_coords], dim=-1)  # Line 334
        assert r_noisy.shape[-1] == 6, "In ECSI, the last dimension should be 6"
```

**Risk Analysis:**
- Doubles the last dimension from 3 to 6
- Example: 16 × 20 × 10000 × 6 × 4 bytes = 76.8 MB
- Old r_noisy tensor should be garbage collected

---

## Issue #7 🟡 MEDIUM-HIGH: Unbounded Random Sampling (Lines 382-393)

**Location:** `sample_noise_level()` method

```python
        if self.logit_normal_sampling:
            # LogitNormal(0, 1) sampling
            # 🟡 MEDIUM-HIGH RISK: Unbounded random tensor creation
            # Memory: batch_size × num_diffusion_samples × 4 bytes
            # No validation on input dimensions
            y = torch.randn(shape, device=device)  # Line 382
            t = torch.sigmoid(y)                    # Line 383
        else:
            # Beta sampling (default to Uniform if alpha=1, beta=1)
            if self.sampling_alpha == 1.0 and self.sampling_beta == 1.0:
                # 🟡 MEDIUM-HIGH RISK: Same issue - unbounded dimensions
                t = torch.rand(shape, device=device)  # Line 387
            else:
                # 🟢 MEDIUM RISK: Beta distribution sampling
                # Can be memory-intensive for large shapes
                m = torch.distributions.Beta(              # Lines 389-393
                    torch.tensor(self.sampling_alpha, device=device),
                    torch.tensor(self.sampling_beta, device=device),
                )
                t = m.sample(shape)
```

**Risk Analysis:**
- No upper bounds on `batch_size` or `num_diffusion_samples`
- Example worst case: 1000 × 1000 = 4 MB (manageable but unbounded)
- Should add validation in Config class

---

## Issue #10 🟢 MEDIUM: Intermediate Computation (Line 519)

**Location:** `interpolate()` method

```python
        # Compute interpolation coefficients
        alpha_t = self.alpha(t_expanded)  # weight for x_0 (holo)
        beta_t = self.beta(t_expanded)    # weight for x_T (apo)
        gamma_t = self.gamma(t_expanded)  # noise scale

        # Mean of bridge distribution: \mu_t = \alpha_t x_0 + \beta_t x_T
        # 🟢 MEDIUM RISK: Creates intermediate tensors
        # Memory: (B, N, La, 3) per intermediate
        # Multiple allocations in single expression
        mu_t = alpha_t * x_holo + beta_t * x_apo  # Line 519
```

**Risk Analysis:**
- Creates 2-3 intermediate tensors for weighted sum
- Each intermediate: B × N × La × 3 × 4 bytes = ~38.4 MB typical
- Part of gradient computation so not immediately freed

---

## Issue #2 🟠 HIGH: Random Noise in Interpolation (Line 522)

**Location:** `interpolate()` method

```python
        # Mean of bridge distribution: \mu_t = \alpha_t x_0 + \beta_t x_T
        mu_t = alpha_t * x_holo + beta_t * x_apo

        # Sample from bridge distribution
        # 🟠 HIGH RISK: Full-sized random tensor allocation
        # Memory: (B, N, La, 3) = ~38.4 MB for typical config
        # Called every training iteration, no size validation
        # Example: 16×20×10000×3×4 = 38.4 MB
        noise = torch.randn_like(x_apo)             # Line 522
        noised_coords = mu_t + gamma_t * noise      # Line 523

        # Mask out padding atoms
        noised_coords = noised_coords * mask[:, None, :, None]
```

**Risk Analysis:**
- Creates full coordinate-sized random tensor
- Called during every training iteration
- No memory checks or limits
- Critical for large proteins (>5000 atoms)

---

## Issue #3 🟠 HIGH: Uncontrolled Prior Sampling (Line 638)

**Location:** `sample_structure()` method

```python
        atom_mask = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)

        # Sample x_T from prior (apo structures)
        # 🟠 HIGH RISK: Unvalidated prior sampling
        # Memory: (B, num_diffusion_samples, Latom, 3)
        # No limit on num_diffusion_samples parameter
        # Example: 16×50×10000×3×4 = 96 MB
        x_apo = self.sample_prior(f_input, num_diffusion_samples)  # Line 638
```

**Risk Analysis:**
- Allocates based on unvalidated `num_diffusion_samples`
- Can be called with arbitrarily large values
- Should add maximum limit and validation

---

## Issue #4 🟠 HIGH: Tensor Cloning (Line 644)

**Location:** `sample_structure()` method

```python
        sample_out["init_coordinates"] = x_apo
        if self.normalize_coordinate:
            x_apo = x_apo / self.sigma_data_end

        # 🟠 HIGH RISK: Full tensor clone doubles memory
        # Memory: Doubles allocation from x_apo
        # Example: If x_apo is 38.4 MB, total becomes 76.8 MB
        # No memory check before cloning
        x_t = x_apo.clone()  # Line 644
```

**Risk Analysis:**
- Doubles memory usage for large tensors
- Necessary for algorithm but expensive
- Should add memory check before cloning

---

## Issue #1 🔴 CRITICAL: Trajectory Accumulation (Lines 647, 724)

**Location:** `sample_structure()` method

```python
        x_t = x_apo.clone()

        # 🔴 CRITICAL RISK: Unbounded trajectory storage
        # Memory: tensor_size × num_steps (default 200)
        # Example: 38.4 MB × 200 = 7.68 GB
        # No limit on trajectory length
        # Can easily cause OOM for moderate configurations
        if return_traj:
            traj.append(x_t.cpu())  # Line 647

        # Reverse time sampling from t=T toward t=0
        for step_idx in range(num_steps):
            # ... [sampling code] ...
            
            # Apply mask
            x_t = x_t * atom_mask[..., None]

            # 🔴 CRITICAL RISK: Repeated trajectory storage in loop
            # Accumulates num_steps tensors without limit
            if return_traj:
                traj.append(x_t.cpu())  # Line 724
```

**Risk Analysis:**
- **MOST CRITICAL ISSUE** in the codebase
- Stores every intermediate state without limits
- Memory grows linearly with num_steps (default 200)
- Example calculation:
  - Single tensor: 38.4 MB
  - After 200 steps: 7.68 GB
- Even CPU storage can exhaust memory
- Must implement trajectory downsampling or limits

---

## Issue #5 🟠 HIGH: Repeated Zeros Allocation (Line 664)

**Location:** `sample_structure()` method, inside sampling loop

```python
        # Reverse time sampling from t=T toward t=0
        for step_idx in range(num_steps):
            # Apply random augmentation
            x_t, x_apo = self.random_augmentation(x_t, x_apo, mask=atom_mask)

            t_curr = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t_curr

            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]), t_curr, device=x_t.device, dtype=x_t.dtype
            )

            # Get denoised prediction \hat{x}_0
            # 🟠 HIGH RISK: Zeros tensor created in every loop iteration
            # Memory: (B, N, La, 3) per iteration × 200 iterations
            # Should pre-allocate outside loop and reuse
            # Example: 38.4 MB × 200 = 7.68 GB cumulative
            x0_hat = torch.zeros_like(x_t)  # Line 664
            
            for st in range(0, num_diffusion_samples, max_parallel_samples):
                # ... [forward model call] ...
```

**Risk Analysis:**
- Creates new full-sized tensor in every sampling step
- With default 200 steps, allocates 200 tensors
- Could be pre-allocated once and reused
- Contributes to memory fragmentation

---

## Issue #11 🟢 MEDIUM: z_hat Computation (Line 690)

**Location:** `sample_structure()` method, inside sampling loop

```python
            # Compute route coefficients
            alpha_t = self.alpha(t_exp)
            beta_t = self.beta(t_exp)
            gamma_t = self.gamma(t_exp)
            alpha_dot = self.alpha_deriv(t_exp)
            beta_dot = self.beta_deriv(t_exp)
            gamma_dot = self.gamma_deriv(t_exp)

            # Compute \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
            # 🟢 MEDIUM RISK: Complex expression with multiple intermediates
            # Memory: 3-5 temporary tensors of size (B, N, La, 3)
            # Called 200 times in loop
            z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)  # Line 690
```

**Risk Analysis:**
- Multiple intermediate allocations per expression
- Each temporary tensor: ~38.4 MB
- Repeated 200 times in sampling loop
- Could optimize with in-place operations

---

## Issue #12 🟢 MEDIUM: Drift Computation (Lines 710-714)

**Location:** `sample_structure()` method, inside sampling loop

```python
                eps_t = self.eta * (
                    gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2
                )

                # Compute drift: b(t) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
                #                     + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
                # 🟢 MEDIUM RISK: Complex multi-line expression
                # Memory: 5+ temporary tensors of size (B, N, La, 3)
                # Each intermediate ~38.4 MB
                drift = (                                    # Lines 710-714
                    alpha_dot * x0_hat
                    + beta_dot * x_apo
                    + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
                )
```

**Risk Analysis:**
- Complex expression creates multiple intermediates
- Each term allocates temporary memory
- Could use fused operations for efficiency

---

## Issue #6 🟠 HIGH: Random Noise in Loop (Line 717)

**Location:** `sample_structure()` method, inside sampling loop

```python
                drift = (
                    alpha_dot * x0_hat
                    + beta_dot * x_apo
                    + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
                )

                # Euler step: x_{t+dt} = x_t + b_t * dt + \sqrt{2\epsilon_t |dt|} * noise
                # 🟠 HIGH RISK: Random noise generated in every iteration
                # Memory: (B, N, La, 3) per iteration
                # Example: 38.4 MB × 200 iterations = 7.68 GB cumulative
                # Should pre-allocate buffer and reuse
                noise = torch.randn_like(x_t)                # Line 717
                # 🟢 MEDIUM RISK: Intermediate computation
                diffusion_scale = torch.sqrt(2 * torch.abs(eps_t) * abs(dt) + 1e-8)  # Line 718
                x_t = x_t + drift * dt + diffusion_scale * noise
```

**Risk Analysis:**
- Generates random tensor in every sampling step
- With 200 steps, creates 200 full-sized tensors
- Similar to Issue #5 but for random tensors
- Could use pre-allocated buffer with in-place generation

---

## Summary Statistics

**File:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`
**Total Lines:** 734
**Issues Found:** 14

### By Risk Level
- 🔴 CRITICAL: 1 issue (Trajectory accumulation)
- 🟠 HIGH: 5 issues (Noise generation, prior sampling, cloning, loop allocations)
- 🟡 MEDIUM-HIGH: 1 issue (Unbounded sampling)
- 🟢 MEDIUM: 7 issues (Various intermediate computations)

### By Location
- `sample_structure()` method: 8 issues (most critical)
- `interpolate()` method: 2 issues
- `forward_model()` method: 1 issue
- `sample_noise_level()` method: 3 issues

### Peak Memory Estimate
**Worst Case Configuration:**
- B=16, N=20, La=10000, steps=200, return_traj=True
- **Total: ~8 GB**

**Realistic Configuration:**
- B=4, N=5, La=2000, steps=100, return_traj=False
- **Total: ~3.4 MB** (manageable)

---

## Immediate Actions Required

1. **Fix trajectory storage** (Line 647, 724) - Add limits or downsampling
2. **Pre-allocate loop tensors** (Lines 664, 717) - Reuse instead of recreating
3. **Add configuration validation** - Limit max values for B, N, La
4. **Document memory requirements** - Add to docstrings and README

---

*End of Annotated Source*
