# POROUS: Porous Media Multiphase Flow Simulator

**POROUS**는 다공성 매질에서의 다상 유동을 시뮬레이션하는 Python 기반 수치해석 프로그램입니다. TOUGH2와 유사한 Integral Finite Difference Method (IFDM)를 사용하며, 물-공기(등온) 및 물-증기(비등온) 시스템을 지원합니다.

## Features

- 1D/2D/3D 정형 격자 지원
- 물-공기 2상 유동 (등온)
- 물-증기 2상 유동 (비등온, 상변화 포함)
- Corey 및 van Genuchten 상대투수율 모델
- van Genuchten 및 Brooks-Corey 모세관압 모델
- Newton-Raphson 완전 음해법 (Fully Implicit)
- 적응적 시간 간격 제어
- VTK/CSV 출력 지원

---

## Directory Structure

```
porous/
├── porous/                    # 메인 패키지
│   ├── __init__.py           # 패키지 초기화
│   ├── core/                 # 핵심 데이터 구조
│   │   ├── mesh.py           # 격자 생성 및 관리
│   │   ├── properties.py     # 암석/유체 물성
│   │   └── state.py          # 상태 변수 관리
│   ├── physics/              # 물리 모델
│   │   ├── flow.py           # Darcy 유동 방정식
│   │   ├── eos.py            # 상태방정식 (물/증기/공기)
│   │   ├── relative_perm.py  # 상대투수율 모델
│   │   └── capillary.py      # 모세관압 모델
│   ├── solver/               # 수치해석 솔버
│   │   ├── newton.py         # Newton-Raphson 솔버
│   │   ├── jacobian.py       # Jacobian 행렬 조립
│   │   └── linear.py         # 선형 시스템 솔버
│   ├── io/                   # 입출력
│   │   ├── input_parser.py   # 입력 파일 파서
│   │   └── output.py         # VTK/CSV 출력
│   ├── utils/                # 유틸리티
│   │   └── constants.py      # 물리 상수 정의
│   └── examples/             # 예제 코드
│       ├── 1d_injection.py
│       ├── water_air_injection.py
│       ├── water_air_2d.py
│       └── heterogeneous_2d.py
├── examples/                  # 예제 문서
├── output/                    # 출력 파일 디렉토리
├── run_example.py            # 메인 실행 스크립트
└── requirements.txt          # 의존성 패키지
```

---

## Background Theory

### 1. Governing Equations

#### 1.1 Mass Conservation (질량 보존)

각 상(phase) β에 대한 질량 보존 방정식:

$$\frac{\partial}{\partial t}(\phi \rho_\beta S_\beta) + \nabla \cdot (\rho_\beta \mathbf{v}_\beta) = q_\beta$$

- $\phi$: 공극률 (porosity)
- $\rho_\beta$: β상의 밀도 [kg/m³]
- $S_\beta$: β상의 포화도 (saturation)
- $\mathbf{v}_\beta$: β상의 Darcy 속도 [m/s]
- $q_\beta$: 소스/싱크 항 [kg/(m³·s)]

#### 1.2 Darcy's Law

각 상의 유동 속도:

$$\mathbf{v}_\beta = -\frac{k \cdot k_{r\beta}}{\mu_\beta}(\nabla P_\beta - \rho_\beta \mathbf{g})$$

- $k$: 절대 투수율 (absolute permeability) [m²]
- $k_{r\beta}$: β상의 상대투수율 (relative permeability)
- $\mu_\beta$: β상의 점성계수 [Pa·s]
- $P_\beta$: β상의 압력 [Pa]
- $\mathbf{g}$: 중력 가속도 벡터 [m/s²]

#### 1.3 Capillary Pressure (모세관압)

$$P_c = P_g - P_l$$

기체상 압력과 액체상 압력의 차이로 정의됩니다.

### 2. Constitutive Relations

#### 2.1 Relative Permeability Models

**Corey Model:**
$$k_{rl} = S_e^{n_l}, \quad k_{rg} = (1-S_e)^{n_g}$$

여기서 유효 포화도:
$$S_e = \frac{S_l - S_{lr}}{1 - S_{lr} - S_{gr}}$$

**van Genuchten-Mualem Model:**
$$k_{rl} = \sqrt{S_e}[1-(1-S_e^{1/m})^m]^2$$

#### 2.2 Capillary Pressure Models

**van Genuchten:**
$$P_c = \frac{1}{\alpha}(S_e^{-1/m} - 1)^{1-m}$$

**Brooks-Corey:**
$$P_c = P_e \cdot S_e^{-1/\lambda}$$

### 3. Numerical Method

#### 3.1 Integral Finite Difference Method (IFDM)

제어 체적 V_n에 대한 적분형 보존 방정식:

$$\frac{d}{dt}\int_{V_n} M dV = \int_{\Gamma_n} \mathbf{F} \cdot \mathbf{n} dA + \int_{V_n} q dV$$

이산화된 형태:
$$V_n \frac{M_n^{n+1} - M_n^n}{\Delta t} = \sum_m A_{nm} F_{nm} + V_n q_n$$

#### 3.2 Newton-Raphson Iteration

비선형 잔차 방정식 R(x) = 0을 풀기 위해:

1. Jacobian 행렬 계산: $J_{ij} = \partial R_i / \partial x_j$
2. 선형 시스템 풀이: $J \cdot \delta x = -R$
3. 변수 갱신: $x^{k+1} = x^k + \delta x$
4. 수렴 조건: $||R|| < \epsilon$ 만족 시 종료

#### 3.3 Time Stepping

적응적 시간 간격 제어:
- Newton 반복 3회 이하: Δt × 2.0
- Newton 반복 5회 이하: Δt × 1.5
- Newton 반복 8회 이상: Δt × 0.8
- 수렴 실패 시: Δt × 0.5 후 재시도

---

## File Descriptions

### Core Module (`porous/core/`)

| 파일 | 설명 |
|------|------|
| `mesh.py` | 격자 생성 및 관리. Cell과 Connection 클래스 정의. 1D/2D/3D 정형 격자 및 방사형 격자 생성 함수 제공. |
| `properties.py` | `RockProperties` (공극률, 투수율, 열물성) 및 `FluidProperties` (밀도, 점성, 압축률) 클래스. 암석 및 유체 물성 데이터베이스 관리. |
| `state.py` | `State` 클래스로 각 셀의 열역학적 상태(압력, 온도, 포화도) 관리. 1차 변수/2차 변수 업데이트 및 상전이 처리. |

### Physics Module (`porous/physics/`)

| 파일 | 설명 |
|------|------|
| `flow.py` | Darcy 법칙 기반 다상 유동 플럭스 계산. 잔차 벡터(residual) 조립. 질량/에너지 보존 방정식 구현. |
| `eos.py` | 상태방정식(EOS). `WaterEOS`, `SteamEOS`, `AirEOS` 클래스. 밀도, 엔탈피, 점성계수 계산. 포화압력/온도 상관식. |
| `relative_perm.py` | 상대투수율 모델. Corey, van Genuchten-Mualem, Linear 모델 구현. 팩토리 함수 제공. |
| `capillary.py` | 모세관압 모델. van Genuchten, Brooks-Corey, Linear 모델 구현. |

### Solver Module (`porous/solver/`)

| 파일 | 설명 |
|------|------|
| `newton.py` | Newton-Raphson 비선형 솔버. `NewtonSolver`, `TimeStepController`, `Simulator` 클래스. 적응적 시간 간격 제어 및 시뮬레이션 드라이버. |
| `jacobian.py` | Jacobian 행렬 조립. 수치 미분을 통한 ∂R/∂x 계산. 희소 행렬(sparse matrix) 형태로 반환. |
| `linear.py` | 선형 시스템 솔버. Direct (LU), GMRES, BiCGSTAB, CG 방법 지원. ILU 전처리기 옵션. |

### I/O Module (`porous/io/`)

| 파일 | 설명 |
|------|------|
| `output.py` | 출력 작성기. `VTKWriter` (ParaView 호환), `CSVWriter`, `TimeHistoryWriter` 클래스. NumPy 저장/로드 함수. |
| `input_parser.py` | 입력 파일 파서. YAML/JSON 형식 지원. |

### Utils Module (`porous/utils/`)

| 파일 | 설명 |
|------|------|
| `constants.py` | 물리 상수 정의. 기체상수, 중력가속도, 물의 임계점 등. 단위 변환 계수. 수치 해석 파라미터 기본값. |

---

## Quick Start

```python
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator
import numpy as np

# 1. Create mesh
mesh = create_mesh_1d(length=100.0, num_cells=50, area=1.0)

# 2. Define rock properties
rock = RockProperties(
    porosity=0.2,
    permeability=np.array([1e-13, 1e-13, 1e-13])
)
rock_properties = [rock] * mesh.num_cells

# 3. Initialize state manager
rel_perm = CoreyRelPerm(slr=0.1, sgr=0.05, nl=2.0, ng=2.0)
cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-4, m=0.45)
state_manager = StateManager(
    mesh.num_cells,
    default_rel_perm=rel_perm,
    default_cap_pressure=cap_pressure
)

# 4. Set initial conditions
state_manager.initialize_uniform(
    pressure=1e6,           # 10 bar
    temperature=293.15,     # 20°C
    liquid_saturation=1.0   # Fully saturated
)

# 5. Set up simulator
simulator = Simulator(mesh, state_manager, rock_properties)
simulator.set_simulation_time(end_time=3600.0, dt_initial=1.0)

# 6. Run simulation
success = simulator.run(verbose=True)

# 7. Get results
results = simulator.get_results()
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **numpy** >= 1.20.0
- **scipy** >= 1.7.0
- **matplotlib** >= 3.4.0 (optional, for visualization)

---

## Running Examples

```bash
# Basic example
python run_example.py

# With plots
python run_example.py --plot

# With VTK output
python run_example.py --vtk --output-dir ./output
```

---

## References

1. Pruess, K., Oldenburg, C., & Moridis, G. (1999). TOUGH2 User's Guide, Version 2.0. Lawrence Berkeley National Laboratory.
2. van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. Soil Science Society of America Journal, 44(5), 892-898.
3. Brooks, R. H., & Corey, A. T. (1964). Hydraulic properties of porous media. Hydrology Papers, Colorado State University.

---

## License

MIT License

## Author

POROUS Development Team
