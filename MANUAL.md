# POROUS Instruction Manual

**POROUS** (**P**article **O**utflow and **R**elaxation **O**f **U**nderground **S**tructures)

지하 구조물 토립자 유실 및 다공성 매질 다상 유동 시뮬레이터 상세 사용 설명서

---

## Table of Contents

1. [설치 및 환경 설정](#1-설치-및-환경-설정)
2. [기본 개념](#2-기본-개념)
3. [시뮬레이션 설정 단계](#3-시뮬레이션-설정-단계)
4. [격자 생성](#4-격자-생성)
5. [물성 정의](#5-물성-정의)
6. [초기 조건 설정](#6-초기-조건-설정)
7. [경계 조건 및 소스항](#7-경계-조건-및-소스항)
8. [솔버 설정](#8-솔버-설정)
9. [결과 출력 및 시각화](#9-결과-출력-및-시각화)
10. [예제 모음](#10-예제-모음)
11. [API Reference](#11-api-reference)
12. [문제 해결](#12-문제-해결)

---

## 1. 설치 및 환경 설정

### 1.1 요구사항

- Python 3.8 이상
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0 (선택, 시각화용)

### 1.2 설치

```bash
# 저장소 클론
git clone https://github.com/JihoeKwon/porousMedia.git
cd porousMedia

# 의존성 설치
pip install -r requirements.txt
```

### 1.3 설치 확인

```python
import porous
print(porous.__version__)  # '0.1.0' 출력
```

---

## 2. 기본 개념

### 2.1 주요 변수 (Primary Variables)

| 유체 시스템 | 상태 | 1차 변수 | 2차 변수 |
|------------|------|---------|---------|
| 물-공기 (등온) | 항상 | 압력 P | 액체 포화도 Sl |
| 물-증기 (비등온) | 단상 액체 | 압력 P | 온도 T |
| 물-증기 (비등온) | 2상 | 압력 P | 액체 포화도 Sl |

### 2.2 단위계

모든 물리량은 SI 단위를 사용합니다:

| 물리량 | 단위 | 비고 |
|-------|------|------|
| 압력 | Pa | 1 bar = 10⁵ Pa |
| 온도 | K | K = °C + 273.15 |
| 길이 | m | |
| 시간 | s | 1 day = 86400 s |
| 투수율 | m² | 1 Darcy = 9.87×10⁻¹³ m² |
| 질량 유량 | kg/s | |

### 2.3 좌표계

- X: 수평 방향 (1D/2D/3D)
- Y: 수평 방향 (2D/3D)
- Z: 연직 방향 (3D), 중력은 -Z 방향

---

## 3. 시뮬레이션 설정 단계

전형적인 시뮬레이션 워크플로우:

```
1. 격자 생성 (Mesh)
       ↓
2. 암석 물성 정의 (RockProperties)
       ↓
3. 상태 관리자 초기화 (StateManager)
       ↓
4. 초기 조건 설정
       ↓
5. 경계 조건/소스항 설정
       ↓
6. 시뮬레이터 생성 및 실행
       ↓
7. 결과 분석 및 시각화
```

---

## 4. 격자 생성

### 4.1 1D 격자

```python
from porous.core.mesh import create_mesh_1d

mesh = create_mesh_1d(
    length=100.0,      # 전체 길이 [m]
    num_cells=50,      # 셀 개수
    area=1.0,          # 단면적 [m²]
    origin=0.0         # 시작 좌표 [m]
)

print(f"셀 수: {mesh.num_cells}")
print(f"연결 수: {mesh.num_connections}")
```

### 4.2 2D 격자

```python
from porous.core.mesh import create_mesh_2d

mesh = create_mesh_2d(
    lx=100.0,          # X방향 길이 [m]
    ly=50.0,           # Y방향 길이 [m]
    nx=20,             # X방향 셀 수
    ny=10,             # Y방향 셀 수
    thickness=1.0,     # Z방향 두께 [m]
    origin=(0.0, 0.0)  # 원점 좌표
)
```

### 4.3 3D 격자

```python
from porous.core.mesh import create_mesh_3d

mesh = create_mesh_3d(
    lx=100.0, ly=50.0, lz=20.0,
    nx=20, ny=10, nz=4,
    origin=(0.0, 0.0, 0.0)
)
```

### 4.4 방사형 격자 (1D)

```python
from porous.core.mesh import create_radial_mesh_1d
import numpy as np

# 로그 간격 방사형 격자
radii = np.logspace(-1, 2, 31)  # 0.1m ~ 100m, 30개 셀
mesh = create_radial_mesh_1d(
    radii=radii,
    height=10.0,       # 원통 높이 [m]
    angle=2*np.pi      # 전체 원주 (360°)
)
```

### 4.5 셀 정보 접근

```python
# 셀 중심 좌표
centers = mesh.get_cell_centers()  # shape: (num_cells, 3)

# 셀 체적
volumes = mesh.get_cell_volumes()  # shape: (num_cells,)

# 특정 셀의 이웃 셀
neighbors = mesh.get_neighbor_cells(cell_index=0)

# 3D 인덱스 변환
cell_idx = mesh.cell_index_3d(i=5, j=3, k=1)  # (i,j,k) → 전역 인덱스
i, j, k = mesh.cell_indices_from_global(cell_idx)  # 전역 인덱스 → (i,j,k)
```

---

## 5. 물성 정의

### 5.1 암석 물성 (RockProperties)

```python
from porous.core.properties import RockProperties
import numpy as np

rock = RockProperties(
    name="sandstone",
    porosity=0.2,                                    # 공극률 [-]
    permeability=np.array([1e-13, 1e-13, 1e-14]),   # 투수율 텐서 [m²]
    density=2650.0,                                  # 암석 밀도 [kg/m³]
    specific_heat=1000.0,                            # 비열 [J/(kg·K)]
    thermal_conductivity=2.0,                        # 열전도율 [W/(m·K)]
    compressibility=0.0,                             # 압축률 [1/Pa]

    # 상대투수율 모델
    rel_perm_model="corey",
    rel_perm_params={
        "slr": 0.1,    # 잔류 액상 포화도
        "sgr": 0.05,   # 잔류 기상 포화도
        "nl": 2.0,     # 액상 Corey 지수
        "ng": 2.0,     # 기상 Corey 지수
    },

    # 모세관압 모델
    cap_pressure_model="van_genuchten",
    cap_pressure_params={
        "slr": 0.1,
        "alpha": 1e-4,   # [1/Pa]
        "m": 0.45,
        "pmax": 1e7,     # 최대 모세관압 [Pa]
    }
)
```

### 5.2 투수율 단위 변환

```python
from porous.core.properties import RockProperties

# millidarcy 단위로 입력
rock = RockProperties.from_dict({
    'name': 'shale',
    'porosity': 0.05,
    'permeability_md': 0.1,  # 0.1 mD → 자동 변환
})

# Darcy 단위로 입력
rock = RockProperties.from_dict({
    'name': 'gravel',
    'porosity': 0.3,
    'permeability_darcy': 10.0,  # 10 D → 자동 변환
})
```

### 5.3 편의 함수

```python
from porous.core.properties import (
    create_rock_with_corey_model,
    create_rock_with_van_genuchten
)

# Corey 모델 사용
rock = create_rock_with_corey_model(
    name="sand",
    porosity=0.25,
    permeability=1e-12,
    slr=0.15, sgr=0.1, nl=3.0, ng=2.5
)

# van Genuchten 모델 사용
rock = create_rock_with_van_genuchten(
    name="clay",
    porosity=0.4,
    permeability=1e-15,
    slr=0.2, alpha=5e-5, m=0.5
)
```

### 5.4 비균질 물성 (Heterogeneous)

```python
# 셀별로 다른 암석 물성 지정
rock_properties = []
for i in range(mesh.num_cells):
    x = mesh.cells[i].center[0]

    if x < 50.0:
        # 고투수율 영역
        rock = RockProperties(porosity=0.25, permeability=np.array([1e-12]*3))
    else:
        # 저투수율 영역
        rock = RockProperties(porosity=0.15, permeability=np.array([1e-14]*3))

    rock_properties.append(rock)
```

---

## 6. 초기 조건 설정

### 6.1 StateManager 초기화

```python
from porous.core.state import StateManager, FluidSystem
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary

# 상대투수율/모세관압 모델 생성
rel_perm = CoreyRelPerm(slr=0.1, sgr=0.05, nl=2.0, ng=2.0)
cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-4, m=0.45)

# StateManager 생성
state_manager = StateManager(
    num_cells=mesh.num_cells,
    default_rel_perm=rel_perm,
    default_cap_pressure=cap_pressure,
    fluid_system=FluidSystem.WATER_AIR  # 또는 FluidSystem.WATER_STEAM
)
```

### 6.2 균일 초기 조건

```python
state_manager.initialize_uniform(
    pressure=1e6,           # 10 bar
    temperature=293.15,     # 20°C (등온 시스템에서도 필요)
    liquid_saturation=1.0   # 완전 포화
)
```

### 6.3 비균일 초기 조건

```python
import numpy as np

# 배열로 초기 조건 설정
n = mesh.num_cells
pressure = np.linspace(2e6, 1e6, n)      # 압력 구배
temperature = np.ones(n) * 293.15         # 균일 온도
saturation = np.linspace(1.0, 0.3, n)    # 포화도 구배

state_manager.initialize_from_arrays(
    pressure=pressure,
    temperature=temperature,
    liquid_saturation=saturation
)
```

### 6.4 개별 셀 상태 접근

```python
# 특정 셀 상태 읽기
state = state_manager.states[0]
print(f"압력: {state.pressure} Pa")
print(f"온도: {state.temperature} K")
print(f"액상 포화도: {state.liquid_saturation}")
print(f"액상 밀도: {state.liquid_density} kg/m³")
print(f"액상 점성: {state.liquid_viscosity} Pa·s")

# 전체 배열로 읽기
pressures = state_manager.get_pressures()
saturations = state_manager.get_liquid_saturations()
temperatures = state_manager.get_temperatures()
```

---

## 7. 경계 조건 및 소스항

### 7.1 대용량 셀 방식 (Large Volume Cell)

압력 또는 포화도 고정 경계 조건 구현:

```python
# 마지막 셀을 일정 압력 경계로 설정
mesh.cells[-1].volume *= 1e8  # 매우 큰 체적

# 첫 번째 셀을 주입 경계로 설정 (일정 포화도)
mesh.cells[0].volume *= 1e8
state_manager.states[0].set_primary_variables(
    pressure=2e6,           # 고정 압력
    second_var=1.0          # 고정 포화도
)
```

### 7.2 소스/싱크 항

```python
import numpy as np

# 소스 항 배열 생성 (num_cells × num_equations)
num_eq = 2  # 물 질량, 공기 질량
source_terms = np.zeros((mesh.num_cells, num_eq))

# 셀 0에 물 주입 (양의 값 = 주입)
source_terms[0, 0] = 0.01  # 0.01 kg/s 물 주입

# 마지막 셀에서 물 생산 (음의 값 = 생산)
source_terms[-1, 0] = -0.01  # 0.01 kg/s 물 생산

# 셀 10에 공기 주입
source_terms[10, 1] = 0.001  # 0.001 kg/s 공기 주입
```

### 7.3 시간 의존적 소스항

```python
class TimeVaryingSource:
    def __init__(self, mesh):
        self.mesh = mesh

    def get_source(self, time):
        source = np.zeros((self.mesh.num_cells, 2))

        # 처음 1시간 동안만 주입
        if time < 3600.0:
            source[0, 0] = 0.01

        return source

# 시뮬레이션 루프에서 사용
time_source = TimeVaryingSource(mesh)
# simulator.source_terms를 매 시간 스텝마다 업데이트
```

---

## 8. 솔버 설정

### 8.1 Simulator 생성

```python
from porous.solver.newton import Simulator

simulator = Simulator(
    mesh=mesh,
    state_manager=state_manager,
    rock_properties=rock_properties,
    include_energy=False,      # 에너지 방정식 포함 여부
    log_file="output/simulation_log.csv"  # 로그 파일 경로
)
```

### 8.2 시간 설정

```python
simulator.set_simulation_time(
    end_time=86400.0,     # 1일 (초 단위)
    dt_initial=1.0        # 초기 시간 간격
)

# 소스 항 설정
simulator.set_source_terms(source_terms)
```

### 8.3 Newton 솔버 파라미터

```python
# NewtonSolver 직접 설정
simulator.newton_solver.tolerance = 1e-5      # 수렴 허용치
simulator.newton_solver.max_iterations = 30   # 최대 반복 횟수
simulator.newton_solver.max_dp = 1e6          # 최대 압력 변화량 [Pa]
simulator.newton_solver.max_dT = 10.0         # 최대 온도 변화량 [K]
simulator.newton_solver.max_dS = 0.2          # 최대 포화도 변화량
```

### 8.4 시간 간격 제어

```python
# TimeStepController 설정
simulator.time_controller.dt_min = 1e-6       # 최소 시간 간격 [s]
simulator.time_controller.dt_max = 86400.0    # 최대 시간 간격 [s]
simulator.time_controller.max_cuts = 10       # 최대 시간 간격 축소 횟수
```

### 8.5 시뮬레이션 실행

```python
success = simulator.run(verbose=True)

if success:
    print("시뮬레이션 성공!")
else:
    print("시뮬레이션 실패!")
```

---

## 9. 결과 출력 및 시각화

### 9.1 결과 데이터 접근

```python
results = simulator.get_results()

times = results['times']            # shape: (num_timesteps,)
pressures = results['pressures']    # shape: (num_timesteps, num_cells)
saturations = results['saturations']
temperatures = results['temperatures']

# 특정 시점 데이터
final_pressure = pressures[-1]      # 마지막 시점
initial_pressure = pressures[0]     # 초기 시점
```

### 9.2 VTK 출력 (ParaView용)

```python
from porous.io.output import VTKWriter

vtk_writer = VTKWriter(output_dir='./output', base_name='simulation')

# 특정 시점 출력
vtk_writer.write(mesh, state_manager, time=simulator.time_controller.current_time)

# 추가 필드 포함
porosity = np.array([rock.porosity for rock in rock_properties])
vtk_writer.write(mesh, state_manager, time=1000.0, Porosity=porosity)
```

### 9.3 CSV 출력

```python
from porous.io.output import CSVWriter

csv_writer = CSVWriter(output_dir='./output', base_name='results')
csv_writer.write(mesh, state_manager, time=1000.0)
```

### 9.4 시계열 출력

```python
from porous.io.output import TimeHistoryWriter

# 특정 셀 모니터링
history_writer = TimeHistoryWriter(
    output_dir='./output',
    filename='monitoring.csv',
    cell_indices=[0, mesh.num_cells//2, mesh.num_cells-1]
)

# 시뮬레이션 루프 내에서 호출
history_writer.write(state_manager, time=current_time)
```

### 9.5 Matplotlib 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

results = simulator.get_results()
x = np.linspace(0, 100, mesh.num_cells)

# 압력 프로파일
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 압력 vs 거리 (여러 시점)
ax = axes[0, 0]
for idx in [0, len(results['times'])//2, -1]:
    t = results['times'][idx]
    ax.plot(x, results['pressures'][idx]/1e5, label=f't={t/3600:.1f}h')
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Pressure [bar]')
ax.legend()
ax.grid(True)

# 2. 포화도 vs 거리
ax = axes[0, 1]
for idx in [0, len(results['times'])//2, -1]:
    t = results['times'][idx]
    ax.plot(x, results['saturations'][idx], label=f't={t/3600:.1f}h')
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Liquid Saturation [-]')
ax.legend()
ax.grid(True)

# 3. 압력 vs 시간 (특정 위치)
ax = axes[1, 0]
for cell in [0, mesh.num_cells//2, mesh.num_cells-1]:
    ax.plot(results['times']/3600, results['pressures'][:, cell]/1e5,
           label=f'x={x[cell]:.0f}m')
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Pressure [bar]')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
```

### 9.6 2D 컨투어 플롯

```python
# 2D 격자 결과 시각화
import matplotlib.pyplot as plt
import numpy as np

nx, ny = mesh.nx, mesh.ny
pressure_2d = results['pressures'][-1].reshape(ny, nx)
saturation_2d = results['saturations'][-1].reshape(ny, nx)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 압력 분포
im1 = axes[0].imshow(pressure_2d/1e5, origin='lower', aspect='auto',
                     extent=[0, 100, 0, 50], cmap='viridis')
axes[0].set_xlabel('X [m]')
axes[0].set_ylabel('Y [m]')
axes[0].set_title('Pressure [bar]')
plt.colorbar(im1, ax=axes[0])

# 포화도 분포
im2 = axes[1].imshow(saturation_2d, origin='lower', aspect='auto',
                     extent=[0, 100, 0, 50], cmap='Blues', vmin=0, vmax=1)
axes[1].set_xlabel('X [m]')
axes[1].set_ylabel('Y [m]')
axes[1].set_title('Liquid Saturation [-]')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('2d_results.png', dpi=150)
```

---

## 10. 예제 모음

### 10.1 1D 물 주입

```python
"""1D 물 주입 시뮬레이션"""
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator
import numpy as np

# 격자
mesh = create_mesh_1d(100.0, 50, 1.0)

# 물성
rock = RockProperties(porosity=0.2, permeability=np.array([1e-13]*3))
rock_properties = [rock] * mesh.num_cells

# 상태 관리자
rel_perm = CoreyRelPerm(slr=0.1, sgr=0.05)
cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-4, m=0.45)
state_manager = StateManager(mesh.num_cells, rel_perm, cap_pressure)

# 초기 조건: 완전 포화
state_manager.initialize_uniform(1e6, 293.15, 1.0)

# 경계 조건: 마지막 셀 고정 압력
mesh.cells[-1].volume *= 1e8

# 소스 항: 첫 셀에 물 주입
source = np.zeros((mesh.num_cells, 2))
source[0, 0] = 0.001  # kg/s

# 시뮬레이션
simulator = Simulator(mesh, state_manager, rock_properties)
simulator.set_simulation_time(3600*6, 0.1)
simulator.set_source_terms(source)
simulator.run()
```

### 10.2 Buckley-Leverett 문제

```python
"""Buckley-Leverett 충격파 문제"""
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import NoCapillary
from porous.solver.newton import Simulator
import numpy as np

# 격자 (100m, 100개 셀)
mesh = create_mesh_1d(100.0, 100, 1.0)

# 물성 (모세관압 무시)
rock = RockProperties(porosity=0.2, permeability=np.array([1e-12]*3))
rock_properties = [rock] * mesh.num_cells

# 상대투수율 (Corey)
rel_perm = CoreyRelPerm(slr=0.2, sgr=0.2, nl=2.0, ng=2.0)

# 모세관압 없음
from porous.physics.capillary import NoCapillary
cap_pressure = NoCapillary()

state_manager = StateManager(mesh.num_cells, rel_perm, cap_pressure)

# 초기 조건: 잔류 물 포화도
state_manager.initialize_uniform(1e6, 293.15, 0.2)

# 경계: 입구 완전 포화, 출구 고정 압력
mesh.cells[0].volume *= 1e8
state_manager.states[0].set_primary_variables(1.5e6, 1.0)
mesh.cells[-1].volume *= 1e8

# 시뮬레이션
simulator = Simulator(mesh, state_manager, rock_properties)
simulator.set_simulation_time(86400, 1.0)  # 1일
simulator.run()
```

### 10.3 5-Spot 패턴 (2D)

```python
"""Five-spot 물 주입 패턴"""
from porous.core.mesh import create_mesh_2d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator
import numpy as np

# 2D 격자 (100m x 100m)
mesh = create_mesh_2d(100.0, 100.0, 20, 20, thickness=1.0)

# 물성
rock = RockProperties(porosity=0.2, permeability=np.array([1e-13]*3))
rock_properties = [rock] * mesh.num_cells

# 상태 관리자
rel_perm = CoreyRelPerm(slr=0.2, sgr=0.2)
cap_pressure = VanGenuchtenCapillary()
state_manager = StateManager(mesh.num_cells, rel_perm, cap_pressure)

# 초기 조건: 공기 포화
state_manager.initialize_uniform(1e6, 293.15, 0.2)

# 주입정 (중앙)
nx, ny = 20, 20
center_cell = ny//2 * nx + nx//2
mesh.cells[center_cell].volume *= 1e6

# 생산정 (네 모서리)
corners = [0, nx-1, (ny-1)*nx, (ny-1)*nx + nx-1]
for c in corners:
    mesh.cells[c].volume *= 1e6
    state_manager.states[c].set_primary_variables(0.5e6, 0.2)

# 소스 항: 중앙에 물 주입
source = np.zeros((mesh.num_cells, 2))
source[center_cell, 0] = 0.01  # 0.01 kg/s

# 시뮬레이션
simulator = Simulator(mesh, state_manager, rock_properties)
simulator.set_simulation_time(86400*30, 10.0)  # 30일
simulator.set_source_terms(source)
simulator.run()
```

---

## 11. API Reference

### 11.1 porous.core.mesh

| 함수/클래스 | 설명 |
|------------|------|
| `Cell` | 단일 셀 데이터클래스 (index, volume, center, rock_type) |
| `Connection` | 셀 간 연결 데이터클래스 (cell1, cell2, area, distance) |
| `Mesh` | 격자 관리 클래스 |
| `create_mesh_1d(length, num_cells, area)` | 1D 격자 생성 |
| `create_mesh_2d(lx, ly, nx, ny, thickness)` | 2D 격자 생성 |
| `create_mesh_3d(lx, ly, lz, nx, ny, nz)` | 3D 격자 생성 |
| `create_radial_mesh_1d(radii, height, angle)` | 방사형 격자 생성 |

### 11.2 porous.core.properties

| 함수/클래스 | 설명 |
|------------|------|
| `RockProperties` | 암석 물성 데이터클래스 |
| `FluidProperties` | 유체 물성 데이터클래스 |
| `MaterialDatabase` | 물성 데이터베이스 |
| `create_rock_with_corey_model()` | Corey 모델 암석 생성 |
| `create_rock_with_van_genuchten()` | VG 모델 암석 생성 |

### 11.3 porous.core.state

| 함수/클래스 | 설명 |
|------------|------|
| `PhaseState` | 상태 열거형 (SINGLE_LIQUID, SINGLE_GAS, TWO_PHASE) |
| `FluidSystem` | 유체계 열거형 (WATER_STEAM, WATER_AIR) |
| `State` | 셀 상태 데이터클래스 |
| `StateManager` | 전체 상태 관리 클래스 |

### 11.4 porous.physics.relative_perm

| 함수/클래스 | 설명 |
|------------|------|
| `RelativePermeability` | 추상 기본 클래스 |
| `CoreyRelPerm` | Corey 모델 |
| `VanGenuchtenRelPerm` | van Genuchten-Mualem 모델 |
| `LinearRelPerm` | 선형 모델 |
| `create_relative_permeability(name, params)` | 팩토리 함수 |

### 11.5 porous.physics.capillary

| 함수/클래스 | 설명 |
|------------|------|
| `CapillaryPressure` | 추상 기본 클래스 |
| `VanGenuchtenCapillary` | van Genuchten 모델 |
| `BrooksCoreyCapillary` | Brooks-Corey 모델 |
| `LinearCapillary` | 선형 모델 |
| `NoCapillary` | 모세관압 없음 |
| `create_capillary_pressure(name, params)` | 팩토리 함수 |

### 11.6 porous.solver.newton

| 함수/클래스 | 설명 |
|------------|------|
| `NewtonSolver` | Newton-Raphson 비선형 솔버 |
| `TimeStepController` | 적응적 시간 간격 제어 |
| `Simulator` | 시뮬레이션 드라이버 |
| `NewtonResult` | Newton 반복 결과 |
| `TimeStepResult` | 시간 스텝 결과 |

---

## 12. 문제 해결

### 12.1 수렴 실패

**증상:** "Time step failed" 또는 "Newton diverged" 메시지

**해결책:**
1. 초기 시간 간격 축소: `dt_initial=0.01`
2. Newton 허용치 완화: `tolerance=1e-4`
3. 변수 변화 제한 완화:
   ```python
   solver.max_dp = 5e5
   solver.max_dS = 0.1
   ```
4. 격자 해상도 증가
5. 소스/싱크 강도 감소

### 12.2 비물리적 결과

**증상:** 음수 압력, 포화도 > 1 또는 < 0

**해결책:**
1. 경계 조건 확인
2. 소스 항 부호 확인 (양수 = 주입)
3. 물성값 범위 확인
4. 시간 간격 축소

### 12.3 느린 수렴

**증상:** Newton 반복 횟수가 항상 최대치

**해결책:**
1. Jacobian 섭동 크기 조정:
   ```python
   assembler.dp = 1000.0   # 압력 섭동
   assembler.dS = 1e-5     # 포화도 섭동
   ```
2. 전처리기 변경: `solver.preconditioner='jacobi'`
3. 선형 솔버 변경: `solver.solver_type=SolverType.GMRES`

### 12.4 메모리 부족

**증상:** MemoryError

**해결책:**
1. 격자 크기 축소
2. 희소 행렬 솔버 사용 (기본값)
3. 출력 빈도 감소

### 12.5 일반적인 디버깅

```python
# 시뮬레이션 로그 확인
with open('output/simulation_log.csv', 'r') as f:
    print(f.read())

# 잔차 모니터링
result = simulator.newton_solver.solve(dt=1.0)
print(f"잔차 norm: {result.residual_norm}")
print(f"최대 잔차: {result.max_residual}")

# 상태 변수 검사
for i, state in enumerate(state_manager.states):
    if state.pressure < 0 or state.liquid_saturation < 0:
        print(f"셀 {i}: 비물리적 상태")
```

---

## Appendix: 물리 상수

| 상수 | 값 | 단위 |
|------|-----|------|
| 기체 상수 R | 8.314 | J/(mol·K) |
| 중력 가속도 | 9.807 | m/s² |
| 물 분자량 | 18.015 | g/mol |
| 물 임계온도 | 647.1 | K |
| 물 임계압력 | 22.06 | MPa |
| 1 Darcy | 9.87×10⁻¹³ | m² |
| 1 bar | 10⁵ | Pa |
| 1 atm | 101,325 | Pa |

---

## Author

**권지회 (Jihoe Kwon)**
한국지질자원연구원 AI융합연구실
Korea Institute of Geoscience and Mineral Resources (KIGAM)
AI Convergence Research Lab

Copyright (c) 2025 권지회, 한국지질자원연구원. All rights reserved.

---

*POROUS v0.1.0 - Particle Outflow and Relaxation Of Underground Structures*
