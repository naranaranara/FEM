# FEM · RIMLS Contact

RIMLS 접촉 에너지와 Hessian 근사 방법을 연구하는 독립 Python FEM 코드입니다.

## 포함된 기능

- 3D 선형 사면체 FEM과 희소 강성 조립
- RIMLS 표면·Gap 및 PyTorch 자동미분
- Barrier 접촉 에너지와 접촉력
- Full Newton, GN, filtered GN, 다중 접촉 Gap SR1
- IPC Toolkit BVH를 이용한 두 물체 접촉 후보 탐색 및 삼각형 CCD 보폭 제한
- 두 변형체의 준정적 압축과 헤르츠 접촉 해석해 비교

Gap SR1은 접촉점별 Gap Gradient 변화에서 곡률을 추정합니다. 기본적으로 최근 수용된 secant pair 5개를 국부 자유도에 보관하며, 접촉 해제·대상 변경·지지 자유도 변경 시 기록을 초기화합니다. 에너지와 접촉력은 기존 식으로 계산하고 Hessian에만 보정을 추가합니다.

## 설치

저장소 루트에서 가상환경을 활성화한 후 실행합니다.

```bash
python -m pip install -e ".[collision,visualization]"
```

개발 시 사용한 환경은 Python 3.8, PyTorch 1.9.0+cpu, NumPy 1.23.5, SciPy 1.10.1, IPC Toolkit Python 패키지 `ipctk` 1.3.0입니다. 현재 구현은 CPU 기준입니다. 기존 환경의 PyTorch/ipctk 네이티브 라이브러리 충돌을 피하도록 충돌 탐색을 별도 자식 프로세스에서 실행합니다.

## 실행

두 변형체에서 네 방법을 비교하고 애니메이션을 표시합니다.

```bash
python run.py two-body --methods full gn filtered_gn sr1_gap --steps 20 --show
```

Gap SR1만 실행합니다.

```bash
python run.py two-body --methods sr1_gap --sr1-memory 5 --steps 20 --show
```

충돌 탐색 없이 미리 지정된 두 접촉면을 사용하는 작은 비교 예제입니다.

```bash
python run.py two-body --n 2 --collision none --methods gn sr1_gap --steps 4
```

헤르츠 비교 예제입니다.

```bash
python run.py hertz --nxy 17 --nz 9 --show
```

`--show`를 생략하면 수치만 출력합니다. 결과를 저장하려면 새 출력 폴더를 지정합니다.

```bash
python run.py hertz --outdir experiments/hertz_contact/new_run --show
python run.py two-body --methods gn sr1_gap --outdir experiments/two_body/new_run
```

기본 `two-body` 방법 목록은 Full/GN/filtered GN입니다. Gap SR1은 `--methods sr1_gap`으로 선택합니다. 헤르츠 예제는 고정 평면에 대한 Gap이 `s=z`인 정확한 단순화 경로를 사용하며, SR1을 호출하지 않습니다.

## 검증

이 저장소에 포함한 독립 검사를 실행합니다. 자동 접촉·CCD 검사에는 `ipctk`가 필요합니다.

```bash
python -m unittest discover -s tests -v
```

검사 내용은 접촉점 선택과 양쪽 물체의 미분 결합, 자동 탐색, Gap SR1 곡률 복원과 기록 초기화, 평형·반력·비침투 및 헤르츠 비교 구현입니다. 작은 평면 블록에서 Gap SR1은 Full과 같은 평형해를 얻었지만 GN 대비 속도 이득을 확인하지 못했습니다. 곡면 다중 접촉에서의 가속 효과는 추가 검증 대상입니다.

## 구성

```text
src/
  geometry/       메시, 표면 샘플링, RIMLS, 접촉 패치
  mechanics/      선형 Solid FEM
  collision/      IPC Toolkit BVH·CCD 연결
  contact/        Barrier 에너지·미분
  optimization/   Newton, GN, SR1, 선 탐색
  simulation/     두 변형체 압축, 헤르츠 비교
run.py            실행 진입점
pyproject.toml    패키지·의존성 설정
tests/            src만 사용하는 독립 검증
```

`src`는 기존 연구의 `fem2d`, `fem3d`, 별도 `ipc` 코드에 의존하지 않습니다. 이 저장소에는 독립 핵심 코드와 검증 파일을 보관합니다. 과거 실험 자료·결과·참고 논문은 포함하지 않습니다. `src/STRUCTURE.txt`와 `src/PLATFORM.txt`에는 전체 로컬 연구 프로젝트의 역사적 경로가 언급될 수 있습니다.

## 현재 검증 범위

소변형 선형 탄성체의 준정적 접촉을 대상으로 합니다. 일반적인 대변형, 천, self-contact, 완전한 동역학 드라이버는 아직 검증하지 않았습니다. `cloth.py`는 미구현 인터페이스입니다. CCD는 명시적 삼각형과 선형 절점 궤적을 보호하며 재구성된 RIMLS 영면의 무교차를 보장하는 것은 아닙니다.

헤르츠 예제는 구면 하부를 가진 유한 탄성체와 강체 평면의 비교입니다. 메시와 경계조건 오차가 남아 있으므로 해석해와의 완전한 일치를 주장하지 않습니다. SR1 보정 행렬은 희소 행렬로 명시적으로 조립하며, 대규모 matrix-free 확장은 아직 검증하지 않았습니다.

## Three-stage cylinder validation (learning runner)

```powershell
python -B -u run.py cylinder-validation --repeats 3 --steps 8
```

`src/simulation/test_cylinder_validation.py` checks surface zero-set errors, activation-distance sensitivity and a final-load GN / Gap SR1 / Gap Structured BFGS comparison. Results are checkpointed under `experiments/cylinder_validation_results/`. See `experiments/CYLINDER_VALIDATION.md` locally for assumptions and output descriptions. Static condensation is exact for the current constant linear FEM model; the timing excludes preparation and the shared loading path. Radial FEM mesh and domain-size convergence remain unverified. No barrier strength parameter is added.
