# ECSI CPU OOM 분석 보고서 (한국어 요약)

## 🔍 조사 개요

**리포지토리:** SeonghwanSeo/kfold  
**분석 파일:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`  
**ECSI 최초 커밋:** `4cd4f5e` (retry CI/CD)  
**분석 일자:** 2025-12-29  
**발견된 이슈:** 14개

---

## 📊 주요 발견 사항

### 전체 이슈 요약

| 위험도 | 개수 | 설명 |
|--------|------|------|
| 🔴 CRITICAL | 1개 | 즉시 수정 필요 |
| 🟠 HIGH | 5개 | 조만간 수정 필요 |
| 🟡 MEDIUM-HIGH | 1개 | 주목할 만한 문제 |
| 🟢 MEDIUM | 7개 | 최적화 고려 대상 |

---

## 🔴 가장 심각한 3가지 문제

### 1번 이슈: Trajectory 누적 (가장 치명적) ⚠️

**위치:** 647, 724번 줄  
**메소드:** `sample_structure()`  

**문제점:**
- `return_traj=True`일 때 모든 중간 단계를 무제한으로 저장
- 기본 설정으로 200개 스텝 전부 저장
- 최악의 경우 **7.68 GB** 메모리 사용

**해결 방법:**
```python
# Config 클래스에 추가
max_trajectory_samples: int = 50  # 최대 50개만 저장

# sample_structure()에서 수정
if return_traj:
    if len(traj) < self.max_trajectory_samples:
        traj.append(x_t.cpu())
```

---

### 2번 이슈: 샘플링 루프 내 랜덤 노이즈 생성

**위치:** 717번 줄  
**메소드:** `sample_structure()`  

**문제점:**
- 매 반복마다 새로운 38.4 MB 텐서 생성
- 200번 반복 시 누적 **7.68 GB** 메모리 사용

**해결 방법:**
```python
# 루프 전에 미리 할당
noise_buffer = torch.zeros_like(x_t)

# 루프 내에서
noise_buffer.normal_()  # 제자리에서 생성
noise = noise_buffer
```

---

### 3번 이슈: 루프 내 Zeros 텐서 반복 생성

**위치:** 664번 줄  
**메소드:** `sample_structure()`  

**문제점:**
- 매 반복마다 38.4 MB 텐서를 새로 생성
- 200번 반복 시 불필요하게 많은 메모리 할당

**해결 방법:**
```python
# 루프 전에 한 번만 할당
x0_hat = torch.zeros_like(x_t)

# 루프 내에서 재사용 (새로 생성하지 않음)
```

---

## 💾 메모리 사용량 표

| 설정 | Batch | Samples | Atoms | Steps | Traj? | 메모리 사용량 |
|------|-------|---------|-------|-------|-------|---------------|
| 최소 | 1 | 1 | 1000 | 50 | No | ~10 MB ✅ |
| 소형 | 4 | 5 | 2000 | 100 | No | ~340 MB ✅ |
| 중형 | 8 | 10 | 5000 | 200 | No | ~2 GB ⚠️ |
| 대형 | 16 | 20 | 10000 | 200 | No | ~8 GB ❌ |
| **위험** | 16 | 20 | 10000 | 200 | **Yes** | **~16 GB** 🔴 |

**메모리 계산 공식:**
```
단일 텐서 = B × N × L × 3 × 4 bytes
피크 메모리 ≈ 단일 텐서 × (1 + num_steps if return_traj else 5)
```

---

## 📋 전체 이슈 목록

| # | 위험도 | 줄 | 메소드 | 설명 | 메모리 영향 |
|---|--------|-----|---------|------|-------------|
| 1 | 🔴 CRITICAL | 647, 724 | `sample_structure()` | Trajectory 무제한 저장 | 7.68 GB |
| 2 | 🟠 HIGH | 522 | `interpolate()` | 랜덤 노이즈 생성 | 38.4 MB/호출 |
| 3 | 🟠 HIGH | 638 | `sample_structure()` | Prior 샘플링 | 96 MB |
| 4 | 🟠 HIGH | 644 | `sample_structure()` | 텐서 복제 | 2배 메모리 |
| 5 | 🟠 HIGH | 664 | `sample_structure()` | 루프 내 zeros | 7.68 GB |
| 6 | 🟠 HIGH | 717 | `sample_structure()` | 루프 내 noise | 7.68 GB |
| 7 | 🟡 MED-HIGH | 382, 387 | `sample_noise_level()` | 무제한 샘플링 | 가변 |
| 8 | 🟢 MEDIUM | 389 | `sample_noise_level()` | Beta 분포 샘플링 | 가변 |
| 9 | 🟢 MEDIUM | 428 | `get_sampling_schedule()` | 스케줄 텐서 | 40 KB |
| 10 | 🟢 MEDIUM | 519 | `interpolate()` | 중간 계산 | 38.4 MB |
| 11 | 🟢 MEDIUM | 690 | `sample_structure()` | z_hat 계산 | 38.4 MB |
| 12 | 🟢 MEDIUM | 710 | `sample_structure()` | Drift 계산 | 38.4 MB |
| 13 | 🟢 MEDIUM | 334 | `forward_model()` | 텐서 결합 | 76.8 MB |
| 14 | 🟢 MEDIUM | 718 | `sample_structure()` | Diffusion scale | 작음 |

---

## 🚀 권장 조치 사항

### 1단계: 즉시 수정 (이번 주)
- [ ] Trajectory 길이 제한 구현 (#1 이슈)
- [ ] 설정값 검증 추가
- [ ] 메모리 요구사항 문서화
- [ ] 대용량 설정 테스트

### 2단계: 단기 수정 (이번 스프린트)
- [ ] 루프 텐서 사전 할당 (#5, #6 이슈)
- [ ] 메모리 프로파일링 추가
- [ ] 큰 설정값에 대한 경고 구현
- [ ] 메모리 제한 단위 테스트

### 3단계: 중기 개선 (다음 스프린트)
- [ ] 중간 할당 최적화 (#10-14 이슈)
- [ ] 텐서 재사용 전략
- [ ] 대용량 배치 청킹 지원
- [ ] CI/CD 메모리 프로파일링

### 4단계: 장기 계획
- [ ] Gradient checkpointing
- [ ] Mixed precision 지원
- [ ] 디스크 기반 trajectory 저장
- [ ] 고급 메모리 최적화

---

## 📚 상세 문서

모든 내용은 영문 문서에 자세히 기록되어 있습니다:

### 빠른 참조 가이드 (먼저 읽으세요) ⭐
**파일:** `ECSI_OOM_QUICK_REFERENCE.md`
- 상위 3개 치명적 이슈
- 빠른 수정 코드 스니펫
- 메모리 스케일링 테이블
- 단계별 실행 계획

### 종합 분석 보고서
**파일:** `ECSI_CPU_OOM_ANALYSIS.md`
- 14개 이슈 상세 설명
- 위험도 분류
- 메모리 영향 계산
- 최악의 시나리오 분석
- 10가지 권장사항

### 주석 달린 소스 코드
**파일:** `ECSI_ANNOTATED_SOURCE.md`
- 인라인 주석이 달린 코드 섹션
- 시각적 위험도 표시
- 코드별 메모리 영향
- 위치별 권장사항

### 문서 인덱스
**파일:** `ECSI_OOM_INDEX.md`
- 전체 문서 개요
- 빠른 링크
- 구현 체크리스트
- 테스트 가이드

---

## 🔍 분석 방법론

1. **Git 히스토리 분석**
   - ECSI가 처음 커밋된 `4cd4f5e` 확인
   - 이후 변경사항 추적 (변경사항 없음 확인)

2. **정적 코드 분석**
   - 734줄의 ECSI 구현 코드 전체 검토
   - PyTorch 텐서 할당 패턴 분석
   - 메모리 누수 가능성 검토

3. **메모리 영향 계산**
   - 각 텐서 할당의 크기 계산
   - 최악의 시나리오 시뮬레이션
   - 현실적 사용 패턴 분석

4. **위험도 분류**
   - 메모리 영향 크기
   - 발생 빈도
   - 수정 긴급도

---

## 💡 핵심 인사이트

### 메모리 문제의 주요 원인
1. **Trajectory 저장** (가장 심각)
   - 제한 없이 모든 중간 상태 저장
   - 기본 200 스텝 × 텐서 크기 = 수 GB

2. **루프 내 반복 할당**
   - 매 반복마다 새 텐서 생성
   - 사전 할당으로 쉽게 개선 가능

3. **검증 부족**
   - 입력 파라미터 크기 제한 없음
   - 설정값 검증 로직 필요

### 메모리 문제가 발생하는 경우
- Trajectory 저장 활성화 시
- 대형 단백질 구조 (>5000 atoms)
- 높은 diffusion sample 수 (N > 10)
- 큰 배치 크기 (B > 8)

---

## ✅ 즉시 적용 가능한 수정사항

### 수정 #1: Trajectory 제한 (가장 중요)
```python
# Config 클래스에 추가
max_trajectory_samples: int = 50

# sample_structure()에서
if return_traj and len(traj) < self.max_trajectory_samples:
    traj.append(x_t.cpu())
```

### 수정 #2: 텐서 사전 할당
```python
# 루프 전
x0_hat = torch.zeros_like(x_t)
noise_buffer = torch.zeros_like(x_t)

# 루프 내
# x0_hat = torch.zeros_like(x_t)  # 삭제
noise_buffer.normal_()  # 717번 줄 대체
```

### 수정 #3: 설정 검증
```python
def _validate_memory_config(self):
    if self.num_steps > 500:
        raise ValueError(f"num_steps too large: {self.num_steps}")
    if self.num_steps > 200:
        import warnings
        warnings.warn("Large num_steps may cause high memory usage")
```

---

## 📞 문의 및 참고

### 관련 파일
- **소스 코드:** `src/kfold/model/modules/structure_module/kfold_ecsi.py`
- **설정 파일:** `configs/model/module/structure_module/ecsi.yaml`
- **테스트:** `tests/structure_module/test_ecsi_*.py`

### Git 명령어
```bash
# 최초 커밋 확인
git show 4cd4f5e

# ECSI 파일 히스토리
git log --follow -- src/kfold/model/modules/structure_module/kfold_ecsi.py
```

---

## 📈 결론

ECSI 구현에서 **14개의 잠재적 CPU OOM 이슈**를 발견했습니다. 

**가장 치명적인 문제:**
- Trajectory 저장이 최대 **7.68 GB** 사용 가능
- 제한 없이 모든 중간 상태를 메모리에 저장

**해결 우선순위:**
1. 🔴 Trajectory 누적 수정 (즉시)
2. 🟠 루프 내 반복 할당 최적화 (이번 주)
3. 🟡 설정 검증 추가 (이번 주)
4. 🟢 중간 계산 최적화 (추후)

**기대 효과:**
- 메모리 사용량 90% 이상 감소 가능
- 대형 단백질 구조 처리 가능
- OOM 에러 방지

---

**분석 완료일:** 2025-12-29  
**상태:** ✅ 분석 완료  
**다음 단계:** 권장 수정사항 구현

*상세 내용은 영문 문서를 참조하세요.*
