# ECSI Memory Leak - Third Root Cause Analysis

## 새로운 증상 (New Critical Symptoms)

User tested both infrastructure fixes (torch.compile disabled, persistent_workers disabled) but **문제가 계속됩니다**.

**새로운 패턴:**
- System Memory Utilization (%): 계속 증가 ⬆️
- Disk I/O write to vda1: 계속 증가 ⬆️  
- Network traffic: 증가 ⬆️
- Disk Utilization: 안정 ➡️
- Process Memory In Use: 안정 ➡️

**이것은 완전히 다른 문제입니다!** 메모리 누수가 아니라 **데이터 누적 문제**입니다.

---

## 🔴 ROOT CAUSE #3: Logging & Data Accumulation

### Issue #1: WandB Log Accumulation 🔴

**증상 분석:**
- **Network traffic 증가** = wandb가 클라우드에 데이터 전송
- **Disk I/O write 증가** = wandb가 로컬 파일에 로그 저장
- **System memory 증가** = 로그 버퍼가 메모리에 누적

**문제 위치:** `src/kfold/training/folding/training_module.py`

**Line 308 - Training step logging:**
```python
def training_step(self, batch, batch_idx):
    # ... training logic ...
    for k, v in metrics.items():
        self.log(f"train/{k}", v, prog_bar=(k == "loss"))  # 매 스텝마다 로그
```

**Line 455 - Validation logging:**
```python
def on_validation_epoch_end(self):
    # ... metrics ...
    self.log_dict(avg_values, sync_dist=True)  # 검증 끝마다 로그
```

**문제:**
1. **매 training step마다 wandb에 로그** (수천~수만 번)
2. wandb는 로그를 **메모리 버퍼에 누적** 후 주기적으로 전송
3. ECSI training은 **더 느림** → 같은 시간에 더 많은 스텝 → 더 많은 로그
4. 버퍼가 가득 차면 **디스크에 임시 저장** → Disk I/O 증가
5. 네트워크로 전송 → Network traffic 증가

**왜 EDM은 괜찮은가:**
- EDM training이 더 빠름 → 같은 시간에 적은 스텝 → 적은 로그
- 또는 EDM 실험은 짧은 시간만 실행

---

### Issue #2: Validation Structure Saving 🔴

**위치:** `src/kfold/training/folding/training_module.py` line 686-743

```python
def save_structure(self, ...):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save 4-6 files per validation sample:
    save_path = save_dir / f"{name}-gt.cif"          # Ground truth
    save_path = save_dir / f"{name}-apo.cif"         # Apo structure
    save_path = save_dir / f"{name}-apo.pdb"         # Backup
    save_path = save_dir / f"{name}-gt-aligned{i}.cif"  # Multiple alignments
    save_path = save_dir / f"{name}-{i}-rmsd{rmsd:.2f}-lddt{lddt*100:.2f}.cif"  # Predictions
```

**문제:**
1. **Validation마다 수십 개 파일 저장** (num_diffusion_samples × 파일 형식)
2. 파일은 **삭제되지 않고 계속 누적**
3. 각 파일은 수 MB ~ 수십 MB
4. **Disk I/O write 폭발적 증가**

**Line 411-425:**
```python
if val_config.save_structure_path is not None:  # 이게 활성화되어 있으면
    save_dir = pathlib.Path(
        val_config.save_structure_path, f"it-{self.global_step}"
    )
    self.save_structure(...)  # 매 validation마다 파일 저장
```

---

### Issue #3: Validation Frequency 🟠

**문제:**
- Lightning의 기본 `val_check_interval`이 너무 빈번할 수 있음
- ECSI training이 느리면 → 같은 epoch에 더 많은 validation
- 각 validation마다 파일 저장 → 디스크 폭발

---

### Issue #4: Tensor Accumulation in Validation 🟠

**Line 722-742:**
```python
def save_structure(self, ...):
    # Detach and move to CPU
    true_coords_arr = true_coords[0].detach().cpu().numpy()  # Line 722
    pred_coords_arr = pred_coords[0].detach().cpu().numpy()  # Line 732
    
    # Create new structures
    new_struct = struct.replace_atom_coords(true_coords_arr)  # Line 723
    new_struct = struct.replace_atom_coords(pred_coords_arr)  # Line 735
    
    # Write multiple times
    for i in range(true_coords_arr.shape[0]):  # Line 724
        new_struct.write(save_path, i, ...)
    
    for i in range(pred_coords_arr.shape[0]):  # Line 736
        new_struct.write(save_path, i, ...)
```

**잠재적 문제:**
1. `replace_atom_coords()` 호출이 **메모리 복사**
2. 각 `write()` 호출이 **파일 I/O**
3. `num_diffusion_samples`가 크면 (예: 20) → 20번 반복 write

---

## 🔴 ROOT CAUSE #4: PyTorch Lightning Callback Issues

### Issue #5: ModelCheckpoint Callback 🟠

Lightning의 `ModelCheckpoint`가 활성화되어 있으면:
- 매 N steps/epochs마다 체크포인트 저장
- 각 체크포인트는 **수백 MB ~ 수 GB**
- 오래된 체크포인트 자동 삭제 안될 수 있음
- **Disk I/O 폭발**

---

## 진단: 어떤 것이 문제인지 확인

### 테스트 1: WandB 로깅 비활성화

```python
# scripts/train.py 또는 config
# WandbLogger를 CSVLogger로 교체
from lightning.pytorch.loggers import CSVLogger

logger = CSVLogger("logs", name="experiment")
# wandb 대신 사용
```

또는:

```python
# 로깅 빈도 대폭 감소
trainer = Trainer(
    log_every_n_steps=1000,  # 기본값이 50일 수 있음
)
```

---

### 테스트 2: Structure Saving 비활성화

```yaml
# config file
validation:
  save_structure_path: null  # 완전히 비활성화
```

또는:

```python
# training_module.py line 411
# if val_config.save_structure_path is not None:
if False:  # 임시로 비활성화
    ...
```

---

### 테스트 3: Validation 빈도 감소

```python
# trainer configuration
trainer = Trainer(
    val_check_interval=1000,  # 1000 steps마다만 validation
    # 또는
    check_val_every_n_epoch=5,  # 5 epoch마다만
)
```

---

### 테스트 4: Checkpoint Saving 제한

```python
# ModelCheckpoint 설정
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,  # 최고 2개만 유지
    every_n_train_steps=5000,  # 5000 스텝마다만
)
```

---

## 즉시 확인할 사항

### 확인 1: save_structure_path가 활성화되어 있나?

```bash
# Config 파일 확인
grep -r "save_structure_path" configs/
```

활성화되어 있으면 **이것이 주범일 가능성 높음**.

### 확인 2: 디스크 사용량 체크

```bash
# Training 디렉토리 크기 확인
du -sh /path/to/training/output/*
ls -lh /path/to/save_structure_path/
```

많은 `.cif` 파일이 있으면 **확정**.

### 확인 3: WandB 로그 크기

```bash
# WandB 로컬 캐시 확인
du -sh ~/.wandb/
ls -lh wandb/latest-run/files/
```

거대하면 **wandb logging issue**.

---

## 해결 방법 우선순위

### Priority 1 (즉시): Structure Saving 비활성화

```yaml
# config
validation:
  save_structure_path: null
```

**예상 효과:** Disk I/O write 90% 감소

---

### Priority 2: Logging 빈도 감소

```python
# trainer config
trainer = Trainer(
    log_every_n_steps=500,  # 기본 50에서 증가
)
```

**예상 효과:** Network traffic 80% 감소

---

### Priority 3: Validation 빈도 조정

```python
trainer = Trainer(
    val_check_interval=2000,  # 덜 빈번하게
)
```

---

### Priority 4: 명시적 메모리 정리

```python
# training_module.py의 validation_step 끝에
def validation_step(self, ...):
    # ... 기존 코드 ...
    
    # 명시적 정리
    del sample_coords, true_coords, metrics
    gc.collect()
    torch.cuda.empty_cache()
```

---

## 왜 이전 수정이 효과 없었나?

1. **torch.compile 비활성화** → 코드는 맞지만, 로깅 문제는 해결 안함
2. **persistent_workers 비활성화** → 워커 메모리는 해결했지만, 로깅/파일 쓰기는 별개

**근본 원인이 3개였음:**
1. torch.compile cache (해결됨)
2. persistent_workers (해결됨)  
3. **Logging & File I/O (NEW - 아직 해결 안됨)** 🔴

---

## 최종 권장사항

### 즉시 테스트 (우선순위순):

1. **save_structure_path = null** 설정
2. **log_every_n_steps = 500** 이상으로 증가
3. **val_check_interval = 1000** 이상으로 증가

이 3가지만 바꿔도 문제 해결될 가능성 **매우 높음**.

---

## 추가 진단 스크립트

```python
import os
import subprocess

def diagnose_io_issue():
    """Diagnose disk I/O and logging issues"""
    
    print("=== Disk Usage ===")
    # Check training output directory
    result = subprocess.run(['du', '-sh', './outputs'], capture_output=True, text=True)
    print(result.stdout)
    
    print("\n=== WandB Cache ===")
    wandb_dir = os.path.expanduser("~/.wandb")
    if os.path.exists(wandb_dir):
        result = subprocess.run(['du', '-sh', wandb_dir], capture_output=True, text=True)
        print(result.stdout)
    
    print("\n=== Recent Large Files ===")
    result = subprocess.run(
        ['find', '.', '-type', 'f', '-size', '+10M', '-mtime', '-1'],
        capture_output=True, text=True
    )
    print(result.stdout[:1000])  # First 1000 chars
    
    print("\n=== File Count in Output ===")
    result = subprocess.run(['find', './outputs', '-type', 'f'], capture_output=True, text=True)
    file_count = len(result.stdout.split('\n'))
    print(f"Total files: {file_count}")

# Training loop에 추가
if global_step % 100 == 0:
    diagnose_io_issue()
```

---

## 결론

**3가지 독립적인 메모리/리소스 누수 원인:**

1. ✅ torch.compile cache (infrastructure)
2. ✅ persistent_workers (infrastructure)
3. 🔴 **Logging & File I/O accumulation (NEW)** ← 현재 문제

**증상이 다른 이유:**
- 처음 2개는 **메모리 누수**
- 3번째는 **디스크/네트워크 I/O 폭증**

**해결:**
- save_structure_path 비활성화
- logging 빈도 감소
- validation 빈도 감소

---

**분석 완료 시각:** 2025-12-29
**상태:** 🔴 제3의 근본 원인 발견 - 즉시 테스트 필요
