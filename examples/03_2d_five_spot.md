# 예제 3: 2D 5-Spot 패턴 (Five-Spot Pattern)

## 개요

**5-spot 패턴**은 석유 회수 및 지열 시스템에서 가장 널리 사용되는
주입/생산 배치입니다. 대칭성을 이용하여 1/4 영역만 시뮬레이션합니다.

## 물리적 배경

### 5-Spot 배치

```
     ● ─────────────────────────── ●
     │ Injector                    │ Producer
     │                             │
     │                             │
     │             ◎               │
     │          Producer           │
     │                             │
     │                             │
     │ Producer                    │ Injector
     ● ─────────────────────────── ●

     ● = Injector (주입정)
     ◎ = Producer (생산정) - 중앙
```

### 대칭성 이용 (1/4 모델)

```
    ┌───────────────────┐
    │ ●                 │ no flow
    │ Inj               │ boundary
    │                   │
    │                   │
    │                   │
    │               ◎   │
    │              Prod │
    └───────────────────┘
      no flow boundary

    모델 영역 = 전체의 1/4
```

## 문제 설정

### 도메인

| 파라미터 | 값 | 단위 |
|----------|-----|------|
| Lx × Ly | 100 × 100 | m |
| 격자 | 20 × 20 | cells |
| 두께 | 10 | m |

### 입력 조건

| 파라미터 | 값 | 단위 | 설명 |
|----------|-----|------|------|
| **암석 물성** |
| 공극률 (φ) | 0.2 | - | |
| 투과도 (k) | 1×10⁻¹³ | m² | 100 mD, 등방성 |
| **유체 물성** |
| 밀도 | 1000 | kg/m³ | 물 |
| 점도 | 0.001 | Pa·s | |
| **초기 조건** |
| 초기 압력 | 100 | bar | 균일 |
| **경계 조건** |
| 주입정 (0,0) | 0.01 | kg/s | 질량 주입 |
| 생산정 (L,L) | 90 | bar | 고정 압력 |
| 외부 경계 | - | - | No-flow |
| **시뮬레이션** |
| 종료 시간 | 30 | days | |

### 격자 배치

```
    j=20 ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
         │  │  │  │  │  │  │  │  │  │  │
         ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
         │  │  │  │  │  │  │  │  │  │  │
         ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
         │  │  │  │  │  │  │  │  │  │  │
         .  .  .  .  .  .  .  .  .  .  .
         │  │  │  │  │  │  │  │  │  │◎ │← Producer
         ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
     j=0 │● │  │  │  │  │  │  │  │  │  │
         └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
         i=0                          i=20
          ↑
       Injector
```

## 실행 방법

```python
import numpy as np
from porous.core.mesh import create_mesh_2d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.solver.newton import Simulator

# 2D 격자 생성
mesh = create_mesh_2d(
    lx=100.0, ly=100.0,
    nx=20, ny=20,
    thickness=10.0
)
print(f"Cells: {mesh.num_cells}, Connections: {mesh.num_connections}")

# 암석 물성
rock = RockProperties(
    porosity=0.2,
    permeability=np.array([1e-13, 1e-13, 1e-13])
)
rocks = [rock] * mesh.num_cells

# 상태 초기화
state_manager = StateManager(mesh.num_cells)
state_manager.initialize_uniform(
    pressure=1e7,      # 100 bar
    temperature=293.15,
    liquid_saturation=1.0
)

# 소스항 설정
num_eq = 2
source = np.zeros((mesh.num_cells, num_eq))

# 주입정: (0,0) = cell index 0
injector_cell = 0
source[injector_cell, 0] = 0.01  # 0.01 kg/s

# 생산정: (nx-1, ny-1) = cell index (nx*ny - 1)
producer_cell = mesh.num_cells - 1
# 고정 압력 경계로 처리 (대용량 셀)
mesh.cells[producer_cell].volume *= 1e8

# 시뮬레이션
simulator = Simulator(mesh, state_manager, rocks)
simulator.set_simulation_time(30 * 86400, dt_initial=100)  # 30 days
simulator.set_source_terms(source)
simulator.run()
```

## 예상 결과

### 1. 압력 분포 (등고선도)

```
    y
    ↑
    │   HIGH (주입정)
    │     ●━━━━━━━━━━━━┓
    │     ┃            ┃
    │     ┃    →→→     ┃
    │     ┃      ↓     ┃
    │     ┃        ↓   ┃
    │     ┃          ◎ ┃ LOW (생산정)
    │     ┗━━━━━━━━━━━━┛
    └────────────────────→ x
```

**특징:**
- 주입정 주변: 고압 (~110 bar)
- 생산정 주변: 저압 (90 bar, 경계조건)
- 대각선 방향으로 압력 구배 형성

### 2. 유선 (Streamlines)

```
    ●━━━━━━━━━━━━━━━━━━┓
    ┃ ╲               ┃
    ┃   ╲             ┃
    ┃     ╲           ┃
    ┃       ╲         ┃
    ┃         ╲       ┃
    ┃           ╲     ┃
    ┃             ╲   ┃
    ┃               ◎ ┃
    ┗━━━━━━━━━━━━━━━━━┛
```

- 주입정에서 생산정으로 직선 유동 (가장 짧은 경로)
- 코너 영역은 유동 정체 (stagnant zone)

### 3. 정상상태 압력 분포

**이론적 해석해 (무한 매질, 점 소스):**
```
P(r) = P_ref + (Q × μ) / (4π × k × h) × ln(r/r_ref)
```

| 위치 | 거리 (m) | 예상 압력 (bar) |
|------|----------|-----------------|
| 주입정 근처 | 5 | ~115 |
| 중앙 | 70 | ~102 |
| 생산정 | - | 90 (BC) |

### 4. 수치 결과 예시

**압력 프로파일 (대각선 방향):**
```
P [bar]
  120 ┤●
      │ ╲
  110 ┤   ╲
      │     ╲
  100 ┤       ╲
      │         ╲
   90 ┤           ╲◎
      └─────────────────
      0    50   100  141
           r [m]
```

### 5. 질량 수지

| 항목 | 유량 (kg/s) |
|------|-------------|
| 주입 | +0.01 |
| 생산 | -0.01 |
| 누적 오차 | < 0.1% |

## 결과 파일

| 파일명 | 설명 |
|--------|------|
| `pressure_2d.vtk` | ParaView용 압력 분포 |
| `pressure_contour.png` | 압력 등고선도 |
| `streamlines.png` | 유선도 |
| `diagonal_profile.csv` | 대각선 압력 프로파일 |

## ParaView 시각화

```bash
# VTK 파일 생성 후
paraview output/pressure_2d.vtk
```

**시각화 단계:**
1. `pressure_2d.vtk` 열기
2. Filters → Contour → Pressure 선택
3. Filters → Stream Tracer로 유선 추가

## 검증 포인트

### 1. 대칭성 확인

- (x, y)와 (y, x) 위치의 압력이 동일해야 함
- 대각선 기준 대칭

### 2. 질량 보존

```
∫ ρ × ∂P/∂t × c × dV + 주입량 - 생산량 = 0
```

### 3. 해석해 비교

정상상태에서 Theis solution과 비교 가능

## 주의사항

> ⚠️ **격자 해상도**
>
> 주입정/생산정 근처는 압력 구배가 급함.
> Well 모델링 없이는 격자 의존성이 있음.
> Peaceman well model 적용 권장.

> ⚠️ **경계 조건**
>
> 1/4 모델 사용 시 대칭 경계 (no-flow) 필수.
> 잘못된 BC는 비대칭 결과 유발.

## 확장 가능한 변형

1. **이방성 투과도**: kx ≠ ky
2. **불균질 매질**: 투과도 분포 적용
3. **2상 유동**: Water flooding 시뮬레이션
4. **열전달**: 지열 시스템 해석

## 참고문헌

1. Peaceman, D.W. (1978). "Interpretation of Well-Block Pressures in Numerical Reservoir Simulation". SPE Journal.
2. Aziz, K. & Settari, A. (1979). Petroleum Reservoir Simulation. Applied Science Publishers.
