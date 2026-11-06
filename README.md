# Adaptive survey

합성 응답자에게 같은 수의 SP 질문을 제시하고, 질문 선택 정책과 선택확률 예측기를 분리해서 비교하는 Python 패키지입니다.

기본 비교는 **Fixed efficient design과 Non-graph adaptive selection**입니다. 두 방법은 같은 동결 Choice learner에 서로 다른 질문 선택 규칙을 적용합니다. Bayesian/GNN은 추후 같은 `Predictor`/`Policy` 인터페이스로 연결할 수 있습니다.

## 설치

Python 3.10 이상이 필요합니다. 작은 신경망을 CPU에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
```

이미 PyTorch가 설치돼 있다면 `python -m pip install -e '.[neural,dev]'`로 설치할 수 있습니다. 검증·프로토콜·MNL만 사용할 때에는 기본 설치로 충분합니다.

## 처음부터 실행

아래 데이터는 개발용으로 새로 생성한 가상 데이터입니다. 실제 Nemotron 자료나 인간 응답을 포함하지 않습니다.

```bash
python -m adaptive_survey demo --output data/demo --respondents 90
python -m adaptive_survey validate --data data/demo --output outputs/demo-validation.json
python -m adaptive_survey split --data data/demo --output outputs/demo-split.json --train 54 --validation 18 --test 18
python -m adaptive_survey train --data data/demo --split outputs/demo-split.json --scenario mmnl --output checkpoints/demo.pt --ensemble 5 --epochs 20
python -m adaptive_survey evaluate --data data/demo --split outputs/demo-split.json --scenario mmnl --checkpoint checkpoints/demo.pt --output outputs/demo-evaluation
python -m adaptive_survey estimate --data data/demo --split outputs/demo-split.json --evaluation outputs/demo-evaluation --output outputs/demo-population.json
```

산출물이 이미 존재하면 새 경로를 사용해야 합니다. 데모 생성도 기존 입력 파일을 덮어쓰지 않습니다.

## 기존 연동 데이터 사용

별도 제공된 파일을 로컬 입력 디렉터리에 배치합니다. 원본 파일은 복사 또는 이름 변경해서 사용하고 저장소에 올리지 않습니다.

| 로컬 이름 | 제공 자료 |
|---|---|
| `profiles.csv` | `nemotron_mapped.csv` |
| `tasks.json` | `sp_tasks.json` |
| `responses.csv` | `synthetic_responses_all_contexts.csv` 또는 전체 응답 CSV |

실제 문항 JSON처럼 Context 필드가 없어도 block과 time code에서 Context를 도출하고 일치 여부를 검사합니다. 프로필 CSV의 UTF-8 BOM도 지원합니다. 응답 CSV는 다음 열을 사용합니다.

```text
uuid,context,task_id,block,time_code,mnl_choice,mmnl_choice
```

300명 샘플을 사용하는 예시는 다음과 같습니다.

```bash
python -m adaptive_survey validate --data data/real --output outputs/real-validation.json
python -m adaptive_survey split --data data/real --output outputs/real-split.json --train 180 --validation 60 --test 60
python -m adaptive_survey train --data data/real --split outputs/real-split.json --scenario mmnl --output checkpoints/real.pt
python -m adaptive_survey evaluate --data data/real --split outputs/real-split.json --scenario mmnl --checkpoint checkpoints/real.pt --output outputs/real-evaluation
```

같은 UUID의 모든 Context와 MNL/MMNL 응답은 같은 split에 배정합니다. `split`은 정확한 인원수를 hash 순서로 정하고 원본 세 파일의 SHA-256과 프로토콜을 저장합니다. 학습·평가는 이 manifest와 파일·체크포인트의 일치를 확인합니다.

큰 CSV의 검증은 응답자별로 연속된 행을 스트리밍합니다. 이미 처리한 UUID가 다시 나오면 실패합니다. `train`과 `evaluate`는 manifest에 지정된 UUID만 메모리에 남깁니다. 전체 패널을 실수로 한 번에 로드하는 것을 막는 retention limit도 있습니다.

## 평가 조건

| 항목 | 기본값 |
|---|---|
| 문항은행 | 54개, Context마다 18개 |
| 초기 공통 질문 | 블록마다 2개, 총 4개 |
| 총 질문 예산 | 6·9·12·15·18개, 초기 4개 포함 |
| 기본 비교 | Fixed efficient / Non-graph adaptive (`fixed` / `weighted_js`) |
| 지표 | NLL, accuracy, Brier, Context macro, paired UUID bootstrap |

예산은 **응답자 한 명의 Context 하나**에 적용합니다. 예산 6은 초기 4문항을 받은 뒤 2문항을 더 받는 조건입니다. 세 Context 사이에서는 이력을 초기화합니다.

`evaluate`는 목적이 다른 두 설문 경로를 별도로 실행합니다. 각 경로 안에서 두 방법의 초기 문항·후보·생성 응답이 같습니다.

| 실행 경로 | 질문 후보 | 평가용 holdout | 예산 | 산출물 |
|---|---:|---:|---|---|
| `prediction` | 14개 | 블록마다 2개, 총 4개 | 6·9·12 | 숨긴 4문항의 NLL·accuracy·Brier |
| `preference` | 18개 | 없음 | 6·9·12·15·18 | 수집 응답과 공통 MNL의 전체 응답 대비 계수 일치도 |

18개 중 4개를 숨긴 채 18개를 질문할 수는 없습니다. 예산 15·18의 holdout 예측은 계산하지 않으며, `summary.json`의 `unavailable_budgets`에 남깁니다. 한 경로의 예산별 이력은 같은 trajectory의 prefix입니다. 다른 경로의 응답이나 다른 평가 응답자의 상태를 공유하지 않습니다.

```bash
# 진단: 초기 문항 전, 초기 4문항 후, 후보 14문항 전부를 받은 상태
python -m adaptive_survey evaluate --data data/demo --split outputs/demo-split.json --checkpoint checkpoints/demo.pt --output outputs/boundaries --limit 3 --tracks prediction --budgets 0 4 14
```

Fixed는 train 응답으로 MNL을 추정한 뒤 train 프로필 분포에서 문항별 예상 Fisher 정보를 평균합니다. 공통 초기 4문항의 정보를 먼저 더하고, 실제 후보 집합에서 정규화된 Fisher 행렬의 log-determinant를 가장 크게 늘리는 문항을 차례로 넣습니다. 전체 순서는 test 답변을 읽기 전에 확정합니다. 가용 후보와 초기 문항을 계획에 반영하므로, 전역 순위를 만든 뒤 holdout을 지우는 기존 방식과 결과가 다를 수 있습니다.

18개 문항은행 안에서 길이별 고정 prefix를 구성하는 **정규화된 local D-optimal greedy 설계**입니다. 식별 가능한 정보 방향, ridge, train 프로필 표본 수, 고정 순서와 prefix별 log-determinant를 기록합니다. 원래 9문항 블록이나 전역 최적해와의 관계는 [설계 문서](docs/DESIGN.md)에 설명했습니다.

## 결과와 비교

`evaluate`는 아래 파일을 생성합니다.

- `traces.jsonl`: 실행 경로, 선택 문항·공개 응답·후보·획득 점수·선택/갱신 시간
- `predictions.jsonl`: prediction 경로의 보호된 문항에 대한 예측확률, 정답은 미포함
- `collections.jsonl`: preference 경로의 예산별 수집 문항, 예측 점수는 미포함
- `fixed-plans.json`: test 응답 공개 전 계획한 Fixed 순서와 prefix별 설계 기준값
- `scores.json`: evaluator가 계산한 응답자별 점수
- `summary.json`: 경로별 가능·불가능 예산, Context/예산별 결과, 정책 차이, seed·파일 hash·Fixed prior·계산 시간

Fixed에도 Non-graph와 **같은 checkpoint**를 사용합니다. 학습 자료, 특징, 네트워크, 손실, validation 기준과 평가 중 가중치를 같게 두므로 기본 비교는 문항 선택 규칙의 차이입니다. Fixed가 답변을 읽어 예측을 갱신하더라도 다음 질문의 순서는 바뀌지 않습니다. 주 paired 비교는 `weighted_js − fixed` NLL이며 UUID 단위로 bootstrap합니다. Random/entropy는 `--policies`로 지정하는 보조 진단에 남겨뒀습니다.

별도로 학습한 Non-graph checkpoint를 `--common-checkpoint`로 주면 모든 정책의 수집 이력을 같은 예측기로 읽는 결과도 기록합니다. 같은 파일을 사용했다면 결과의 `common_reader_distinct_checkpoint`가 false입니다. 다른 checkpoint라도 같은 모델 계열이므로 모형 중립적인 정보량을 측정하는 것은 아닙니다.

## Non-graph

- 전체 문항 속성 52개, Context 3개, 선택 one-hot 4개를 event MLP에 넣습니다: 59→64→64.
- 이력을 mean pool하고 프로필 9개·Context·이력 개수·빈 이력 표시를 결합합니다: 78→64→32.
- 후보 전체 choice set도 52→64→32로 인코딩합니다.
- 개인 상태, 대안 속성, A/B/C/NONE 라벨, Context, 후보 set embedding을 공유 head에 넣습니다: 84→64→1. 네 효용에 softmax를 적용합니다.
- 모델당 26,113개 파라미터, 기본 ensemble 5개입니다. Bootstrap 단위는 UUID이며 모든 Context를 함께 재표집합니다.
- 하나의 learner가 두 경로의 무작위 부분 이력을 함께 학습합니다. Prediction 경로는 길이 0..14, preference 경로는 0..17이며, 길이가 충분하면 해당 경로의 공통 초기 4문항을 포함합니다. 예측 target은 이력 밖 문항입니다. Prediction holdout은 그 경로의 학습 이력에도 넣지 않습니다. Validation bank는 전체 validation cohort와 두 경로에서 고정 seed로 추출하고 표본 수·경로·hash를 기록합니다.
- 기본 학습 상한은 120 epochs × 1,024 episodes, batch 64로 모델당 1,920 optimizer updates입니다. Validation 720 episodes와 patience 30으로 선택하고, 선택된 모델과 실제 실행의 update 수를 각각 기록합니다. 짧은 실행은 이력 사용을 거의 학습하지 못할 수 있습니다.
- 평가 중 가중치는 동결합니다. 실제 또는 가상 답변은 immutable history에 추가되고, 가상 예측은 한 번에 batch 처리합니다.

UUID·task ID는 모델 feature가 아닙니다. 현재 문항은행은 학습에서도 볼 수 있으므로 결과는 새 응답자 일반화에 관한 것입니다. 순서가 없는 generator에 맞춰 이력 순서를 사용하지 않습니다.

## 질문 선택의 해석

Weighted JS는 후보 q의 네 가상 답변을 넣었을 때 **다른 남은 후보의 예측이 얼마나 달라지는지** 계산합니다. 예상 답변 확률로 가중한 JS 평균이며 질문 자신과 holdout을 제외합니다.

점수는 0부터 log(4) 사이지만 실제 generator에 대한 정확한 Bayesian 정보이득이나 test NLL 감소를 보장하지 않습니다. 일반 ensemble 간 차이를 개인 latent의 불확실성으로 취급하지도 않습니다. `weighted_js_scores(..., diagnostics=True)`로 총확률 일관성의 TV 진단을 얻을 수 있습니다.

MMNL은 개인의 지속적인 랜덤 선호를 가지므로 주 개인화 환경입니다. MNL은 프로필 조건부 개인화의 대조 환경입니다. 이력·정책의 실제 효과는 Random 및 profile-only 결과와 비교해야 합니다. 작은 smoke run을 성능 향상의 증거로 보고하지 않습니다.

Non-graph 명세와 현재 네트워크의 대응 관계는 [설계 문서](docs/DESIGN.md#non-graph-specification)를 참고하세요. [이전 개발 실행 기록](docs/RECHECK.md)은 초기 2문항·예산 3·5·7·9를 사용한 과거 버전의 기록입니다. 현재 기본 설정의 실험 결과로 사용하지 않습니다.

## 공통 MNL과 대체 예측기

`estimate`는 기본적으로 **preference 경로**의 수집 응답과 동일 응답자의 전체 18문항 reference를 공통 pooled MNL로 분석합니다. 실제 handoff와 같은 34개 mean-utility feature를 사용하고, reference와 selected fit의 식별 basis를 맞춥니다. 부족한 rank를 조용히 0계수로 채워 복원 성공으로 보고하지 않습니다. 예산 18에서는 두 방법이 같은 응답 전체를 수집하므로 reference와 일치해야 합니다.

```bash
python -m adaptive_survey estimate --data data/real --split outputs/real-split.json --evaluation outputs/real-evaluation --output outputs/population.json --cluster-se
```

MMNL 응답에 대한 pooled MNL은 reference agreement입니다. 정책별 문항 분포와 모형 오명세의 영향을 받으며, 실제 MMNL 평균계수 복원과 같지 않습니다. 모집단 적합은 사후 분석이고 질문 선택에 되돌리지 않습니다.

```bash
python -m adaptive_survey fit-baseline --data data/demo --split outputs/demo-split.json --output checkpoints/panel.json
python -m adaptive_survey evaluate --data data/demo --split outputs/demo-split.json --predictor panel_map --checkpoint checkpoints/panel.json --policies fixed random entropy --output outputs/panel-evaluation --limit 3
```

PanelMAP은 train에서 추정한 선형 선택모형과 개인 Gaussian MAP 갱신을 쓰는 보조 예측기입니다. 팀의 Bayesian adaptive design 구현과 동일하다고 부르지 않습니다.

## 외부 Generator로 재생성

별도 전달받은 **신뢰할 수 있는** `Response_Generator_Handoff`와 models를 로컬 경로에 둡니다.

```bash
python -m adaptive_survey generate --handoff /local/Response_Generator_Handoff --profiles data/real/profiles.csv --tasks data/real/tasks.json --output outputs/new-responses.csv --limit 300 --seed 2026
```

MMNL taste는 UUID마다 한 번 뽑아 54문항에서 유지합니다. 응답 난수는 UUID·문항·scenario에 묶어 입력 순서와 무관하게 재현됩니다. Generator 코드·모델·공개 입력·출력의 hash를 manifest에 남깁니다. 기존 전달본의 RNG 스크립트와 같은 bytes라고 주장하지 않습니다. 다른 response seed의 민감도는 실제 새 응답표를 만든 뒤 비교해야 합니다.

## 모듈과 남은 범위

| 모듈 | 역할 |
|---|---|
| `schema`, `data`, `protocol` | 공개 타입, 검증, scenario별 응답환경, UUID 분할 |
| `neural`, `training` | Non-graph, bootstrap ensemble, checkpoint |
| `acquisition`, `policies` | 가상 답변 JS와 질문 선택 |
| `runner`, `metrics` | 공통 실행, 예측, 점수, paired CI |
| `estimation`, `baselines` | 공통 MNL, train Fisher 기반 Fixed 설계, 보조 PanelMAP |
| `generator`, `demo`, `cli` | 외부 handoff, 가상 개발자료, 실행 명령 |

GNN/Bayesian native 모델, rollout 이력 보강, 명시적 개인 latent posterior, 자동 hyperparameter bank 실행은 별도 확장입니다. 현재는 hyperparameter를 CLI로 지정해 validation에서 선택할 수 있습니다. [설계·정보 경계](docs/DESIGN.md)와 [실행 검증 기록](docs/VALIDATION.md)을 참고하세요.

실제 프로필·응답·Generator 코드/계수·체크포인트·결과 디렉터리는 이 저장소에 포함하지 않습니다.
