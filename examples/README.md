# POROUS 예제 모음

POROUS 시뮬레이터의 검증 및 사용법 학습을 위한 예제 모음입니다.

## 예제 목록

| 번호 | 예제명 | 파일 | 난이도 | 설명 |
|------|--------|------|--------|------|
| 1 | 1D 물 주입 | `01_1d_water_injection.md` | ★☆☆ | 단일상 Darcy 유동 기본 예제 |
| 2 | 1D Buckley-Leverett | `02_1d_buckley_leverett.md` | ★★☆ | 2상 유동 해석해 비교 (이론) |
| 3 | 2D 5-spot 패턴 | `03_2d_five_spot.md` | ★★☆ | 석유공학 표준 벤치마크 |
| 4 | 1D 물-공기 주입 | `04_1d_water_air_injection.md` | ★★☆ | 등온 2상 유동 (Buckley-Leverett 유형) |

## 실행 방법

```bash
# 단일상 예제 실행
python run_example.py --plot

# 2상 물-공기 예제 실행
python -m porous.examples.water_air_injection

# 옵션
python -m porous.examples.water_air_injection --no-plot  # 그래프 없이 실행
python -m porous.examples.water_air_injection --quiet    # 간략 출력
```

## 결과 확인

모든 결과는 `output/` 폴더에 저장됩니다:
- `*.png`: 시각화 그래프
- `*.vtk`: ParaView용 3D 데이터
- `*.csv`: 수치 데이터

## 유체 시스템

POROUS는 두 가지 유체 시스템을 지원합니다:

| 시스템 | 설명 | 예제 |
|--------|------|------|
| `WATER_AIR` | 등온 물-공기 2상 | 예제 4 |
| `WATER_STEAM` | 열역학적 물-증기 | 예제 1 (단일상) |
