# 예제 4: 1D 물-공기 주입 (Water-Air Injection)

## 개요

다공성 매질에서의 **등온 2상 유동(isothermal two-phase flow)**을 시뮬레이션합니다.
물을 공기로 포화된 다공성 매질에 주입하여 Buckley-Leverett 유형의 변위(displacement)를 재현합니다.

## 물리적 배경

### 지배 방정식

**물 질량 보존:**
```
∂(φρw·Sw)/∂t + ∇·(ρw·vw) = qw
```

**공기 질량 보존:**
```
∂(φρa·Sa)/∂t + ∇·(ρa·va) = qa
```

**Darcy 법칙 (각 상):**
```
vα = -(k·krα/μα)·∇Pα
```

여기서:
- Sw, Sa: 물, 공기 포화도 (Sw + Sa = 1)
- krw, kra: 물, 공기 상대투과도
- ρw, ρa: 물, 공기 밀도

### Buckley-Leverett 이론

**분율 유동 함수 (Fractional Flow):**
```
fw = 1 / (1 + (kra/μa) / (krw/μw))
```

**충격파 조건 (Welge 접선):**
```
fw(Swf) / (Swf - Swr) = dfw/dSw|Swf
```

## 문제 설정

### 도메인

```
┌─────────────────────────────────────────────────────────┐
│  물 주입                                   공기 포화     │
│  (Water)                                   (Air)        │
│     ↓                                        ↓          │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐     │
│  │██│██│██│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│░░│BC│     │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘     │
│  x=0    front →                            x=100m       │
│  ██ = 물 포화    ░░ = 공기 포화                          │
└─────────────────────────────────────────────────────────┘
```

### 입력 조건

| 파라미터 | 값 | 단위 | 설명 |
|----------|-----|------|------|
| **도메인** |
| 길이 (L) | 100 | m | 1D 도메인 길이 |
| 셀 개수 | 100 | - | 충격파 해상도를 위해 증가 |
| 단면적 (A) | 1.0 | m² | |
| **암석 물성** |
| 공극률 (φ) | 0.2 | - | |
| 투과도 (k) | 1×10⁻¹³ | m² | 100 mD |
| **유체 물성 - 물** |
| 밀도 (ρw) | ~998 | kg/m³ | 20°C |
| 점도 (μw) | 0.001 | Pa·s | |
| **유체 물성 - 공기** |
| 밀도 (ρa) | ~1.2 | kg/m³ | 10 bar, 20°C |
| 점도 (μa) | 1.8×10⁻⁵ | Pa·s | |
| **상대투과도 (Corey)** |
| 잔류 물 포화도 (Swr) | 0.2 | - | |
| 잔류 공기 포화도 (Sar) | 0.2 | - | |
| 물 지수 (nw) | 2.0 | - | |
| 공기 지수 (na) | 2.0 | - | |
| **초기 조건** |
| 초기 물 포화도 | 0.2 | - | = Swr (잔류 물) |
| 초기 압력 | 10 | bar | |
| 초기 온도 | 20 | °C | 고정 (등온) |
| **경계 조건** |
| 입구 (x=0) | 0.01 | kg/s | 물 질량 주입률 |
| 출구 (x=L) | 10 | bar | 고정 압력 |
| **시뮬레이션** |
| 종료 시간 | 10 | days | |

### 상대투과도 곡선

**Corey 모델:**
```
krw = ((Sw - Swr) / (1 - Swr - Sar))^nw
kra = ((1 - Sw - Sar) / (1 - Swr - Sar))^na
```

```
    krw, kra
       │
   1.0 ┤        ╱── kra
       │       ╱
       │      ╱
   0.5 ┤     ╱    ╲
       │    ╱      ╲── krw
       │   ╱        ╲
   0.0 ┼──╱──────────╲──────
       Swr          1-Sar    Sw
       0.2           0.8
```

## 실행 방법

### 방법 1: 직접 실행

```bash
cd D:\Claude\porous
python -m porous.examples.water_air_injection
```

### 방법 2: Python 코드

```python
import numpy as np
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager, FluidSystem
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator

# 격자 생성
mesh = create_mesh_1d(length=100.0, num_cells=100, area=1.0)
mesh.cells[-1].volume *= 1e8  # 출구 고정압력 BC

# 암석 물성
rock = RockProperties(
    porosity=0.2,
    permeability=np.array([1e-13, 1e-13, 1e-13])
)
rocks = [rock] * mesh.num_cells

# 상대투과도 모델
rel_perm = CoreyRelPerm(slr=0.2, sgr=0.2, nl=2.0, ng=2.0)
cap_pressure = VanGenuchtenCapillary(slr=0.2, alpha=1e-4, m=0.45)

# 상태 관리자 (물-공기 등온 시스템)
state_manager = StateManager(
    mesh.num_cells,
    default_rel_perm=rel_perm,
    default_cap_pressure=cap_pressure,
    fluid_system=FluidSystem.WATER_AIR
)

# 초기 조건: 잔류 물 + 공기 포화
state_manager.initialize_uniform(
    pressure=1e6,           # 10 bar
    temperature=293.15,     # 20°C (고정)
    liquid_saturation=0.2   # Swr
)

# 소스항
source = np.zeros((mesh.num_cells, 2))
source[0, 0] = 0.01  # 물 주입 10 g/s

# 시뮬레이션
simulator = Simulator(mesh, state_manager, rocks)
simulator.set_simulation_time(end_time=10*86400, dt_initial=10.0)
simulator.set_source_terms(source)
simulator.run(verbose=True)
```

## 예상 결과

### 1. 포화도 프로파일 (거리 vs 물 포화도)

```
    Sw
     │
 0.8 ┤████████████████████████████████████
     │                                    │
     │                                    │ 급격한 변화
 Swf ┤                                    └─────────
     │                                            │
 Swr ┤                                            ░░░░░
     └──────────────────────────────────────────────────
     0            front                           L  x
```

**특징:**
- 입구 근처: Sw ≈ 0.8 (최대 이동 포화도)
- 충격파(shock front)에서 급격한 포화도 변화
- 충격파 후방: Sw = Swr = 0.2

### 2. 시뮬레이션 결과 (10일 후)

| 항목 | 값 |
|------|-----|
| 압력 범위 | 10.0 - 58.2 bar |
| 물 포화도 범위 | 0.200 - 0.957 |
| 전선(front) 위치 | ~60.5 m |

### 3. 분율 유동 곡선

```
    fw
     │
 1.0 ┤                    ╭────────
     │                  ╱
     │                ╱
 0.5 ┤              ╱
     │            ╱
     │          ╱
 0.0 ┼────────╱──────────────────────
     Swr    Swf                1-Sar  Sw
     0.2   0.55                 0.8
```

**Welge 접선 방법에 의한 이론값:**
- 충격파 포화도 (Swf) ≈ 0.55
- 충격파에서 분율 유동 (fw) ≈ 0.9

### 4. 압력 분포

```
P [bar]
  60 ┤●
     │ ╲
  50 ┤   ╲
     │     ╲
  40 ┤       ╲
     │         ╲
  30 ┤           ╲
     │             ╲
  20 ┤               ╲
     │                 ╲
  10 ┤                   ╲______◎
     └─────────────────────────────
     0        50        100     x [m]
```

## 물-공기 vs 물-증기 시스템

| 특성 | 물-공기 (WATER_AIR) | 물-증기 (WATER_STEAM) |
|------|---------------------|----------------------|
| 온도 | 고정 (등온) | 변동 (열역학적 평형) |
| 주변수 | (P, Sw) 항상 | (P, T) 또는 (P, Sw) |
| 상변화 | 없음 | 있음 (증발/응축) |
| 가스상 | 공기 (비압축성) | 증기 (포화 조건) |
| 용도 | Buckley-Leverett | 지열, 증기 주입 |

## 결과 파일

| 파일명 | 설명 |
|--------|------|
| `water_air_saturation.png` | 포화도 및 압력 분포 |

## 검증 포인트

### 1. 질량 보존

```
총 주입량 = 주입률 × 시간
저장량 증가 = Σ(φ × ΔSw × ρw × V)
```

### 2. 전선 속도

**이론적 전선 속도:**
```
v_front = qt / (φ × A) × dfw/dSw|Swf
```

### 3. Buckley-Leverett 해석해 비교

- 충격파 위치 비교
- 포화도 프로파일 형태 비교

## 수치적 고려사항

### 격자 해상도

- 충격파 포착을 위해 최소 50-100개 셀 권장
- 셀 수 증가 시 충격파가 더 선명해짐

### 시간 스텝

- Newton 수렴을 위해 적절한 초기 시간 스텝 필요
- 급격한 포화도 변화 시 시간 스텝 자동 감소

### 수치 확산

- 저차 업윈드 스킴 사용으로 충격파가 약간 퍼짐
- TVD 스킴 적용 시 개선 가능

## 확장 가능한 변형

1. **점도비 변화**: μw/μa 비율 변경
2. **이방성 투과도**: kx ≠ ky
3. **불균질 매질**: 투과도 분포 적용
4. **2D 확장**: 5-spot 패턴 등

## 참고문헌

1. Buckley, S.E. & Leverett, M.C. (1942). "Mechanism of Fluid Displacement in Sands". Trans. AIME.
2. Welge, H.J. (1952). "A Simplified Method for Computing Oil Recovery by Gas or Water Drive". Trans. AIME.
3. Lake, L.W. (1989). Enhanced Oil Recovery. Prentice Hall.
