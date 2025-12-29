# ECSI Memory Leak - Additional Investigation

## 새로운 정보 (New Information)

**증상 (Symptoms):**
- wandb 로그의 process memory in use: ~0.4 (일정하게 유지)
- 전체 시스템 메모리: 계속 감소
- EDM 설정: 문제 없음
- ECSI + PairMixer: 메모리 누수 발생

## 추가로 발견한 잠재적 원인들

### 🔴 CRITICAL #1: torch.compile() 캐시 누수

**위치:** `src/kfold/model/modules/trunk/pairmixer_trunk.py` line 85-87

```python
def do_compile(self):
    self.pairmixer_module = torch.compile(
        self.pairmixer_module, dynamic=False, fullgraph=False
    )
```

**문제:**
1. `torch.compile()`은 컴파일된 코드를 내부 캐시에 저장
2. ECSI의 복잡한 텐서 그래프가 컴파일될 때 더 많은 메타데이터 저장
3. `dynamic=False`로 설정되어 입력 shape 변경 시 재컴파일
4. 컴파일 캐시가 시스템 메모리에 누적되지만 프로세스 메모리로 추적 안됨

**왜 EDM은 괜찮은가:**
- EDM의 단순한 ODE 연산은 컴파일 캐시가 작음
- ECSI의 15+ intermediate 텐서는 더 큰 컴파일 아티팩트 생성

**해결책:**
```python
# Option 1: torch.compile 비활성화 테스트
def do_compile(self):
    pass  # Disable compilation to test

# Option 2: 컴파일 캐시 주기적으로 클리어
import torch._dynamo
torch._dynamo.reset()  # Clear compilation cache

# Option 3: dynamic=True로 변경
self.pairmixer_module = torch.compile(
    self.pairmixer_module, dynamic=True, fullgraph=False
)
```

---

### 🔴 CRITICAL #2: persistent_workers DataLoader 메모리 누수

**위치:** `src/kfold/training/folding/dataset/datamodule.py` line 269, 295

```python
DataLoader(
    ...,
    num_workers=self.config.num_workers,
    persistent_workers=True if self.config.num_workers > 0 else False,
)
```

**문제:**
1. `persistent_workers=True`는 워커 프로세스를 재사용
2. 각 워커가 메모리를 누적적으로 보유
3. 워커 프로세스의 메모리는 메인 프로세스 RSS에 포함 안됨
4. ECSI의 복잡한 데이터 전처리가 워커 메모리에 누적

**왜 EDM은 괜찮은가:**
- 동일한 DataLoader 사용하지만, ECSI training이 더 메모리 집약적
- ECSI의 prior sampling, interpolation이 더 많은 메모리 사용

**해결책:**
```python
# Option 1: persistent_workers 비활성화
DataLoader(
    ...,
    num_workers=self.config.num_workers,
    persistent_workers=False,  # Force worker restart
)

# Option 2: num_workers=0 (싱글 프로세스)
# 성능 저하되지만 메모리 누수 테스트 가능
```

---

### 🟠 HIGH #3: @lru_cache 누적

**위치:** 여러 파일에 분산

```python
# src/kfold/model/modules/sequence_encoder/esmc.py
@lru_cache
def _get_token_to_id() -> dict[str, int]:
    ...

# src/kfold/training/folding/dataset/datamodule.py
@lru_cache
def load_manifest(manifest_path: str | Path) -> list[Metadata]:
    ...

# src/kfold/training/folding/dataset/utils/symmetry.py
@lru_cache(100)
def some_function():
    ...
```

**문제:**
1. `@lru_cache`는 함수 결과를 무한정 캐싱 (maxsize 없으면)
2. 큰 객체 (Metadata 리스트 등)가 메모리에 영구 보존
3. 프로세스 메모리로 추적되지 않을 수 있음 (Python 내부 캐시)

**해결책:**
```python
# 모든 lru_cache에 maxsize 추가
@lru_cache(maxsize=10)  # Limit cache size
def load_manifest(...):
    ...

# 또는 주기적으로 캐시 클리어
from functools import lru_cache
load_manifest.cache_clear()
```

---

### 🟠 HIGH #4: ECSI forward_model의 model_cache 재사용

**위치:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`

```python
# training_step (line 580)
denoised_atom_coords = self.forward_model(
    ...,
    model_cache=model_cache,  # Passed from outside
)

# sample_structure (line 628)
model_cache = {}  # New cache
```

**문제:**
1. `training_step`에서 `model_cache`가 외부에서 전달될 수 있음
2. 캐시가 training loop 전체에서 재사용되면 누적
3. 명시적으로 클리어되지 않음

**해결책:**
```python
# training_step 끝에 추가
def training_step(self, ...):
    result = {
        # ... 기존 코드
    }
    
    # Clear cache if provided
    if model_cache is not None:
        model_cache.clear()
    
    return result
```

---

### 🟡 MEDIUM #5: random_augmentation의 내부 텐서 생성

**위치:** `src/kfold/utils/geometry/random_augment.py` line 210-214

```python
def _center_random_augmentation_torch(...):
    ...
    R = random_rotations_torch(...)  # Line 202 - creates rotation matrix
    ...
    noise = torch.randn(...)  # Line 208 - creates noise
    coords = coords + noise * s_trans  # Line 214
```

**문제:**
1. 매 forward pass마다 rotation matrix와 noise 생성
2. ECSI는 sampling loop에서 200번 호출 (line 652)
3. 생성된 텐서들이 즉시 해제되지 않을 수 있음

**특히 ECSI에서 문제:**
```python
# Line 652 in kfold_ecsi.py
x_t, x_apo = self.random_augmentation(x_t, x_apo, mask=atom_mask)
```
- x_t와 x_apo 모두 augment하므로 2배 메모리 사용
- EDM은 한 번만 호출

---

### 🟡 MEDIUM #6: Recycling Loop의 컴파일된 모듈 호출

**위치:** `src/kfold/model/modules/trunk/pairmixer_trunk.py` line 126-163

```python
for i in range(0, num_recycles + 1):
    enable_grad = self.training and i == num_recycles
    
    with torch.set_grad_enabled(enable_grad):
        if enable_grad and torch.is_autocast_enabled():
            torch.clear_autocast_cache()
        
        # Line 155 - 컴파일된 모듈 호출
        s, z = pairmixer_module(
            s, z,
            mask=f_input.token.pad_mask,
            use_cuequiv_mul=self.use_cuequiv_kernels,
        )
```

**문제:**
1. 각 recycle iteration에서 컴파일된 모듈 호출
2. `torch.clear_autocast_cache()` 호출은 autocast만 클리어
3. 컴파일 캐시는 클리어하지 않음

**해결책:**
```python
for i in range(0, num_recycles + 1):
    ...
    s, z = pairmixer_module(...)
    
    # 명시적으로 중간 텐서 삭제
    if i < num_recycles:
        del s, z  # Will be reassigned next iteration
    
    # 주기적으로 Python GC
    if i % 2 == 0:
        import gc
        gc.collect()
```

---

## 종합 분석: 왜 ECSI만 영향받는가?

### ECSI-specific 메모리 패턴

1. **더 많은 torch.compile 호출:**
   - ECSI의 복잡한 SDE 연산이 더 큰 컴파일 아티팩트 생성
   - 15+ intermediate tensors → larger compiled graphs
   - 200 sampling steps × 복잡한 그래프 = 큰 컴파일 캐시

2. **더 빈번한 augmentation:**
   - `x_t, x_apo = self.random_augmentation(x_t, x_apo, ...)` (line 652)
   - 두 개 텐서를 동시에 augment → 2배 메모리
   - 200번 반복 → 많은 임시 텐서 생성

3. **persistent_workers와의 상호작용:**
   - ECSI training이 워커 프로세스에서 더 많은 메모리 사용
   - prior_coords, label_coords 샘플링이 복잡
   - interpolation 연산이 추가적인 메모리 사용

4. **model_cache 패턴:**
   - ECSI는 `sample_structure`에서 model_cache 사용
   - 200 steps 동안 캐시 누적
   - EDM도 사용하지만 더 간단한 연산

### EDM이 괜찮은 이유

1. **단순한 ODE 연산:**
   - 4개 intermediate tensors만
   - 작은 컴파일 아티팩트
   - 더 빠른 가비지 컬렉션

2. **덜 복잡한 augmentation:**
   - 한 번에 하나의 텐서만 augment
   - 더 적은 임시 텐서

3. **선형 계산 흐름:**
   - torch.compile이 최적화하기 쉬움
   - 캐시 효율성 높음

---

## 실행 가능한 해결책 (우선순위)

### 테스트 1: torch.compile 비활성화
```python
# pairmixer_trunk.py
def do_compile(self):
    pass  # Temporarily disable
```

**예상 결과:** 메모리 누수 크게 감소

---

### 테스트 2: persistent_workers 비활성화
```python
# datamodule.py line 269, 295
DataLoader(
    ...,
    persistent_workers=False,  # Change to False
)
```

**예상 결과:** 워커 메모리 누수 제거

---

### 테스트 3: 명시적 캐시 클리어
```python
# kfold_ecsi.py - sample_structure 끝에
try:
    # ... sampling logic ...
    return sample_out
finally:
    model_cache.clear()
    del model_cache
    import gc
    gc.collect()
```

---

### 테스트 4: 컴파일 캐시 주기적 클리어
```python
# training loop에서
import torch._dynamo

for epoch in range(num_epochs):
    for batch in dataloader:
        # ... training ...
        
        if batch_idx % 100 == 0:
            torch._dynamo.reset()  # Clear compile cache
            gc.collect()
```

---

## 진단 스크립트

```python
import torch
import torch._dynamo
import gc

def diagnose_memory_leak():
    """Diagnose where memory is being held"""
    
    # 1. Check torch.compile cache
    if hasattr(torch._dynamo, 'utils'):
        print("Torch compile cache size:")
        print(torch._dynamo.utils.counters)
    
    # 2. Check CUDA cache (if using GPU)
    if torch.cuda.is_available():
        print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 3. Check Python GC
    gc.collect()
    print(f"Garbage objects: {len(gc.get_objects())}")
    
    # 4. Check lru_cache sizes
    from kfold.training.folding.dataset.datamodule import load_manifest
    print(f"load_manifest cache: {load_manifest.cache_info()}")
    
    return

# Training loop에 추가
if step % 100 == 0:
    diagnose_memory_leak()
```

---

## 권장 조치 순서

1. **즉시 테스트:**
   - torch.compile 비활성화
   - persistent_workers=False 설정
   
2. **효과 확인 후:**
   - 명시적 캐시 클리어 추가
   - lru_cache maxsize 제한
   
3. **최적화:**
   - dynamic=True로 compile 재활성화
   - persistent_workers 조건부 사용
   - 주기적 캐시 클리어

---

## 결론

**핵심 원인:**
1. `torch.compile()` 캐시가 시스템 메모리에 누적
2. `persistent_workers=True`가 워커 메모리 누적
3. ECSI의 복잡한 연산이 두 문제를 증폭

**프로세스 메모리는 안정, 시스템 메모리는 증가하는 이유:**
- 컴파일 캐시와 워커 프로세스 메모리는 메인 프로세스 RSS에 포함 안됨
- wandb는 메인 프로세스만 추적
- 실제 시스템 메모리는 계속 증가

**해결 방법:**
테스트 1과 2를 먼저 시도 → 가장 빠른 효과 예상
