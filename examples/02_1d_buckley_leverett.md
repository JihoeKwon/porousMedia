# 예제 2: 1D Buckley-Leverett 문제

## 개요

Buckley-Leverett 문제는 다공성 매질에서의 **2상 유동(two-phase flow)**을
다루는 고전적인 벤치마크 문제입니다. 해석해가 존재하여 수치 코드 검증에 널리 사용됩니다.

## 물리적 배경

### 문제 설명

- 초기에 오일(또는 가스)로 포화된 1D 다공성 매질
- 한쪽 끝에서 물을 주입
- 물이 오일을 밀어내며 전진 (displacement)
- **충격파(shock front)** 형성

### 지배 방정식

**물 포화도 수송 방정식:**
```
φ ∂Sw/∂t + ∂fw/∂x × (qt/A) = 0
```

**분율 유동 함수 (Fractional Flow):**
```
fw = 1 / (1 + (krw/μw) / (kro/μo))
```

여기서:
- Sw: 물 포화도 [-]
- fw: 물의 분율 유동 [-]
- qt: 총 유량 [m³/s]
- krw, kro: 물, 오일 상대투과도

### Buckley-Leverett 해석해

충격파 위치:
```
xf(t) = (qt × t) / (φ × A) × (dfw/dSw)|Swf
```

충격파 포화도 Swf는 다음 조건에서 결정:
```
fw(Swf) / Swf = (dfw/dSw)|Swf
```

## 문제 설정

### 도메인

```
┌─────────────────────────────────────────────────────────┐
│  물 주입                                    오일 포화   │
│  Sw=1.0                                    Sw=Swr      │
│     ↓                                        ↓          │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                       │
│  │██│░░│░░│░░│░░│░░│░░│░░│░░│░░│                       │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                       │
│  x=0        shock front           x=100m                │
│  ██ = 물    ░░ = 오일                                   │
└─────────────────────────────────────────────────────────┘
```

### 입력 조건

| 파라미터 | 값 | 단위 | 설명 |
|----------|-----|------|------|
| **도메인** |
| 길이 (L) | 100 | m | |
| 셀 개수 | 100 | - | 충격파 해상도 위해 증가 |
| 단면적 (A) | 1.0 | m² | |
| **암석 물성** |
| 공극률 (φ) | 0.2 | - | |
| 투과도 (k) | 1×10⁻¹³ | m² | 100 mD |
| **유체 물성 - 물** |
| 밀도 (ρw) | 1000 | kg/m³ | |
| 점도 (μw) | 0.001 | Pa·s | |
| **유체 물성 - 오일** |
| 밀도 (ρo) | 800 | kg/m³ | |
| 점도 (μo) | 0.005 | Pa·s | 물보다 5배 점성 |
| **상대투과도 (Corey)** |
| 잔류 물 포화도 (Swr) | 0.2 | - | |
| 잔류 오일 포화도 (Sor) | 0.2 | - | |
| 물 지수 (nw) | 2.0 | - | |
| 오일 지수 (no) | 2.0 | - | |
| **초기 조건** |
| 초기 물 포화도 | 0.2 | - | = Swr (잔류 물) |
| 초기 압력 | 100 | bar | |
| **경계 조건** |
| 입구 물 포화도 | 1.0 | - | 순수 물 주입 |
| 주입률 | 1×10⁻⁵ | m³/s | |
| 출구 | - | - | 자유 유출 |
| **시뮬레이션** |
| 종료 시간 | 0.3 | PVI | Pore Volume Injected |

### 상대투과도 모델

**Corey 모델:**
```
krw = krw_max × ((Sw - Swr) / (1 - Swr - Sor))^nw
kro = kro_max × ((1 - Sw - Sor) / (1 - Swr - Sor))^no
```

```
      krw, kro
        │
    1.0 ┤        ╱── kro
        │       ╱
        │      ╱
    0.5 ┤     ╱    ╲
        │    ╱      ╲── krw
        │   ╱        ╲
    0.0 ┼──╱──────────╲──────
        Swr          1-Sor    Sw
```

## 실행 방법

```python
import numpy as np
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.solver.newton import Simulator

# 격자 생성 (고해상도)
mesh = create_mesh_1d(length=100.0, num_cells=100, area=1.0)

# 암석 물성
rock = RockProperties(
    porosity=0.2,
    permeability=np.array([1e-13, 1e-13, 1e-13]),
    rel_perm_model="corey",
    rel_perm_params={
        "slr": 0.2,   # 잔류 물
        "sgr": 0.2,   # 잔류 오일 (여기서는 가스 대신 오일)
        "nl": 2.0,
        "ng": 2.0
    }
)

# 초기 조건: 잔류 물 + 오일 포화
state_manager.initialize_uniform(
    pressure=1e7,       # 100 bar
    temperature=293.15,
    liquid_saturation=0.2  # Swr
)

# 입구 경계: 물 포화도 = 1.0 (순수 물 주입)
# 구현 시 첫 번째 셀을 Sw=1.0으로 고정

# 시뮬레이션 실행
simulator.run()
```

## 예상 결과

### 1. 포화도 프로파일 (거리 vs 물 포화도)

```
    Sw
     │
 1.0 ┤████████████████████
     │                    │
     │                    │ shock front
 Swf ┤                    └─────────────
     │
 Swr ┤                            ░░░░░░░
     └────────────────────────────────────
     0         xf                      L  x
```

**특징:**
- 입구에서 Sw = 1.0 (순수 물)
- **충격파(shock)**: 급격한 포화도 변화
- 충격파 후방: Sw = Swr (잔류 물)
- 충격파는 시간에 따라 오른쪽으로 이동

### 2. 충격파 위치 vs 시간

| 시간 (PVI) | 충격파 위치 (m) | 충격파 포화도 |
|------------|-----------------|---------------|
| 0.1 | ~33 | ~0.55 |
| 0.2 | ~67 | ~0.55 |
| 0.3 | ~100 | ~0.55 |

### 3. 분율 유동 곡선

```
    fw
     │
 1.0 ┤                    ╱────
     │                  ╱
     │                ╱
 0.5 ┤              ╱
     │           ╱
     │        ╱
 0.0 ┼──────╱─────────────────
     Swr                 1-Sor    Sw
```

### 4. 해석해와 비교

**Welge 접선 방법으로 계산한 충격파 포화도:**
```
Swf ≈ 0.55 (Corey n=2, μw/μo = 0.2 인 경우)
```

**검증 기준:**
- 충격파 위치 오차 < 5%
- 포화도 분포 형태 일치
- 질량 보존 오차 < 1%

## 결과 파일

| 파일명 | 설명 |
|--------|------|
| `saturation_profiles.png` | 시간별 포화도 분포 |
| `shock_position.png` | 충격파 위치 vs 시간 |
| `fractional_flow.png` | 분율 유동 곡선 |
| `mass_balance.csv` | 질량 보존 검증 데이터 |

## 수치적 고려사항

### 수치 확산 (Numerical Diffusion)

- 저차 스킴 사용 시 충격파가 퍼짐 (smearing)
- 격자 수 증가로 완화 가능
- 고차 스킴 (TVD, ENO) 권장

### 시간스텝 제한

**CFL 조건:**
```
Δt < Δx × φ × A / (qt × max(dfw/dSw))
```

### 권장 격자 해상도

- 최소 100개 셀 권장
- 충격파 두께 ≈ 2-3 셀

## 참고문헌

1. Buckley, S.E. & Leverett, M.C. (1942). "Mechanism of Fluid Displacement in Sands". Trans. AIME, 146, 107-116.
2. Welge, H.J. (1952). "A Simplified Method for Computing Oil Recovery by Gas or Water Drive". Trans. AIME, 195, 91-98.

## 주의사항

> ℹ️ **물-공기 2상 유동이 구현되었습니다!**
>
> 예제 4 (`04_1d_water_air_injection.md`)에서 Buckley-Leverett 유형의
> 물-공기 2상 유동 시뮬레이션을 실행할 수 있습니다.
>
> ```bash
> python -m porous.examples.water_air_injection
> ```
>
> 물-오일 시스템은 점도비를 조정하여 유사하게 모사할 수 있습니다.
> 완전한 물-오일 EOS는 향후 추가 예정입니다.
