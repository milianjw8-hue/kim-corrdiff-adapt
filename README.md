# kim-corrdiff-adapt (kcadapt)

ERA5 같은 입력으로 학습한 **CorrDiff 다운스케일링 모델에 KIM 입력을 넣을 때** 필요한 준비와 실험을 돕는 도구입니다.

- KIM 입력이 학습 입력(ERA5)과 얼마나 다른지 변수별로 진단합니다.
- 분위수 사상(quantile mapping)으로 KIM 입력 분포를 ERA5에 맞춥니다.
- CorrDiff 정규화 통계(JSON)를 만들고 바꿉니다.
- 자료를 PhysicsNeMo CorrDiff `hrrr_mini` 형식으로 내보냅니다.
- 비교 실험 5가지(R, A, B, C, D)의 PhysicsNeMo 설정 파일을 만듭니다.

GPU 없이 합성 자료로 전체 흐름을 시험할 수 있습니다. 이 저장소에는 **코드와 합성 자료만** 있습니다.
KIM, LDAPS 같은 내부 자료와 학습된 가중치는 넣지 않습니다.

*English summary below.*

## 왜 필요한가

CorrDiff는 입력 채널마다 학습 자료의 평균과 표준편차로 정규화합니다. 입력을 ERA5에서 KIM으로 바꾸면
같은 날씨라도 분석장의 편차, 변동성 차이, 계절별 편향 때문에 입력 분포가 달라집니다.
신경망은 학습 때 본 분포 밖의 입력에 약합니다. 그래서 먼저 차이를 재고, 고칠 방법을 비교해야 합니다.

## 비교 실험

| 실험 | 입력 | 정규화 통계 | 가중치 | 의미 |
|---|---|---|---|---|
| R | ERA5 | 원래 | 원래 | 기준: 학습 때와 같은 입력 |
| A | KIM 그대로 | 원래 | 원래 | 적응 없이 바꿨을 때 |
| B | KIM, 분위수 사상 | 원래 | 원래 | 입력만 보정 (재학습 없음) |
| C | KIM 그대로 | KIM으로 다시 계산 | 원래 | 입력 정규화만 다시 맞춤 (재학습 없음) |
| D | KIM 그대로 | 원래 | 회귀 UNet만 KIM으로 추가 학습 | 확산 모델은 그대로 |

평가는 자매 저장소 [ai-downscale-verify](https://github.com/milianjw8-hue/ai-downscale-verify)로
ASOS/AWS 지점에서 합니다. B와 C는 재학습 없이 바로 쓸 수 있는 방법이고, D는 GPU 학습이 필요합니다.

## 설치

```bash
git clone https://github.com/milianjw8-hue/kim-corrdiff-adapt.git
cd kim-corrdiff-adapt
pip install -e ".[dev]"
```

## 사용 순서

입력 파일은 (time, y, x) 격자의 NetCDF이며, 변수 이름은 CorrDiff 채널 이름과 같아야 합니다.
KIM과 ERA5는 **같은 저해상도 격자**로 미리 맞춰 둡니다.

```bash
# 1) 차이 진단 (KIM과 ERA5가 함께 있는 기간)
kcadapt shift --src kim.nc --ref era5.nc --stats stats_era5.json --out shift.csv --maps shift_maps.nc

# 2) 분위수 사상: 마지막 해는 빼고 맞춘 뒤 그 해로 효과 확인
kcadapt qmap-fit --src kim.nc --ref era5.nc --by season --test-years 2025 --out qmap.nc
kcadapt qmap-apply --qmap qmap.nc --src kim_eval.nc --out kim_eval_qm.nc

# 3) 실험 C용 정규화 통계
kcadapt renorm --stats stats_era5.json --kim kim_train.nc --out stats_kim.json

# 4) CorrDiff 입력 파일 (hrrr_mini 형식)
kcadapt export --input kim_eval_qm.nc --output ldaps_eval.nc --invariant orog_lsm.nc --out eval_kim_qm.nc

# 5) PhysicsNeMo 설정 파일
kcadapt arms --setup configs/setup_example.yaml --out configs/generated
```

설정 파일은 PhysicsNeMo `examples/weather/corrdiff`에서 이렇게 씁니다.

```bash
python generate.py --config-path=/path/to/configs/generated --config-name=config_generate_armB
```

### GPU 없이 시험

```bash
kcadapt demo --out demo
pytest
```

`demo/`에 차이 진단표(`shift_test_year.csv`), 분위수 그림(`qq_test_year.png`), 통계 JSON, 내보낸 자료,
설정 파일이 만들어집니다. 모두 **합성 자료**입니다.

## 주의할 점

- **분위수 사상의 한계:** 변수 하나하나의 분포만 맞춥니다. 공간 구조와 변수 사이 관계는 고치지 않습니다.
  KIM과 ERA5가 함께 있는 기간이 짧으면 격자점별 사상(`pointwise`)이 불안정합니다. 그때는 `--pooled`를 쓰세요.
- **입력 통계 바꾸기(C):** KIM의 체계적 편차를 지우지만, 실제 날씨 차이도 함께 지울 수 있습니다.
- **추가 학습(D):** PhysicsNeMo `train.py`는 체크포인트 폴더의 최신 파일에서 학습을 이어 갑니다.
  사전학습 회귀 체크포인트를 그 폴더에 복사한 뒤 실행합니다. 학습률(2e-5)과 추가 표본 수(200만)는 출발점일 뿐이며 검증 손실을 보고 정합니다(추정).
- **설정 키:** 2026-11-18 PhysicsNeMo 코드 기준입니다. 쓰는 버전과 맞는지 확인하세요.
- 판정 기준(`needs_correction`)의 임계값은 선별용 경험값입니다. 문헌 근거가 있는 값이 아닙니다.

## 방법

자세한 정의는 [`docs/METHODS.md`](docs/METHODS.md)에 있습니다.

## 관련 저장소

- [ai-downscale-verify](https://github.com/milianjw8-hue/ai-downscale-verify): 지점 관측 검증
- [ai-weather-korea-eval](https://github.com/milianjw8-hue/ai-weather-korea-eval): 공개 AI 모델 평가 (초기장 비교 포함)
- [korea-wx-bench](https://github.com/milianjw8-hue/korea-wx-bench): 결과 모음과 결과표

## 라이선스

MIT. PhysicsNeMo와 CorrDiff 가중치는 각자의 라이선스를 따릅니다.

---

## English summary

`kcadapt` prepares and organises the move of a CorrDiff downscaling model trained on ERA5-like inputs
to inputs from KIM (the Korean Integrated Model). It diagnoses the input shift per channel (mean bias,
standard-deviation ratio, normalised 1-Wasserstein distance, shift relative to the CorrDiff training
statistics), fits empirical quantile mapping (pointwise or pooled, by season or month, additive or
multiplicative) with held-out-year evaluation, builds and edits CorrDiff statistics JSON, exports paired
data in the PhysicsNeMo `hrrr_mini` layout, and writes Hydra configs for five arms: ERA5 reference (R),
raw KIM (A), quantile-mapped KIM (B), KIM-renormalised inputs (C) and a regression UNet fine-tuned on KIM
with the diffusion model frozen (D). Station verification is done with ai-downscale-verify. Everything
runs on synthetic data without a GPU; no operational data or weights are included.
