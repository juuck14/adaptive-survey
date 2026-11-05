# Non-graph 재실험

## 재실험 이유

최초 300명 샘플의 실행은 180/60/60명으로 나누고, 모델 5개를 각각
12 epochs × 512 episodes / batch 64로 학습했다. 모델당 최대 optimizer
update가 96회였다. 문항 선택은 달라졌지만 답변 이력이 예측에 거의 영향을
주지 않았으며, 이 실행으로 adaptive 효과를 판단하기에는 학습량이 부족했다.

이후 같은 구조의 단일 모델을 더 학습한 진단에서 이력에 따른 예측 변화가
나타났다. 이 진단에 이미 사용한 기존 test 60명은 재실험의 최종 평가에서
제외했다. 진단 결과를 최종 정책 성능으로 재사용하지 않는다.

## 고정한 조건

| 항목 | 조건 |
|---|---|
| 학습 / 검증 | 기존 180 / 60명, UUID와 응답 유지 |
| 새 시험 표본 | 전체 mapped profile에서 기존 300명을 모두 제외하고 hash로 선택한 300명 |
| 응답 생성 | 별도 제공된 실제 MMNL handoff, seed 2026, UUID마다 한 번 뽑은 taste를 54문항에서 유지 |
| 예측기 | 기존 DeepSets 구조·손실 유지, UUID bootstrap ensemble 5개 |
| 학습 seed | 2026, 2027, 2028; 각 seed마다 독립적인 ensemble 5개 |
| 학습 상한 | 120 epochs, 1,024 episodes/epoch, batch 64 → 모델당 최대 1,920 updates |
| 모델 선택 | 검증 bank 720 episodes, patience 30; test로 선택하지 않음 |
| 평가 | 같은 300명 × 3 Context, 같은 holdout 4개와 초기 질문 2개 |
| 문항 예산 | 초기 질문 포함 3 / 5 / 7 / 9개 |
| 정책 | train Fisher 순위 Fixed, Random, entropy, weighted JS |
| 고정한 평가 난수 | rollout 및 protocol seed 2026, 학습 seed 간 동일 |

학습 seed는 초기화뿐 아니라 UUID bootstrap, 학습 episode, 검증 bank 추출에도
영향을 준다. 따라서 seed 간 표준편차는 이 학습 절차 전체의 변동이다. 각
ensemble 내부의 모델 5개는 같은 검증 bank를 공유한다.

원본 train/validation 응답은 기존 CSV에서 보존했고 새 test 응답만 생성했다.
새 응답은 저장소의 문서화된 RNG 규칙을 사용한다. 원래 CSV와 byte 단위로
같은 난수 생성 절차라는 주장은 하지 않는다. 공개 프로필·문항·응답과 모델
파일의 hash 및 분할·실행 명령은 비공개 실행 manifest에 남긴다.

## 비교 방법

주 비교는 네 예산·세 Context를 같은 비중으로 평균한 **weighted JS − Random
NLL**이다. 음수이면 JS가 유리하다. 각 학습 seed의 동결된 ensemble로 실제
질문을 순차 선택하고, 공개된 답변만 이력에 추가한 뒤 holdout을 예측한다.

3개 ensemble의 예측확률을 다시 합치지 않는다. 각 ensemble의 점수를 계산한
뒤 그 점수를 평균한다. 동일 응답자의 Context·예산·학습 seed 차이를 먼저
평균하고 UUID 300개를 재표집한 paired bootstrap 5,000회로 95% 구간을 구한다.
이는 해당 동결 모델과 한 번의 응답 생성에 조건부인 UUID 불확실성이며,
학습 seed 3개를 독립 응답자 900명으로 세지 않는다. 학습 seed 간 SD는 별도로
보고한다. 예산별·정확도·Fixed 비교는 부차적인 기술 통계다.

이력 사용 여부는 JS가 실제로 수집한 9개 응답을 읽은 예측과 같은 holdout의
빈 이력 예측을 비교한다. 이것이 좋아졌다는 사실만으로 JS가 Random보다
좋은 문항을 골랐다고 판단하지 않는다.

## 정책 비교 결과

새 test 300명, 각 Context의 holdout 4개로 예산마다 3,600개 응답을 평가했다.
아래 값은 3개 학습 seed에서 얻은 점수의 평균과 표본 SD다. 예산 3/5/7/9와
세 Context를 동일 가중했다. NLL은 낮을수록 좋다.

| 정책 | 평균 NLL ± 학습 seed SD | 평균 정확도 |
|---|---:|---:|
| Fixed | 1.12578 ± 0.00183 | 46.139% |
| Random | 1.12616 ± 0.00160 | 46.146% |
| Entropy | 1.12487 ± 0.00117 | 46.123% |
| Weighted JS | 1.12388 ± 0.00191 | 46.181% |

주 비교 **JS − Random = −0.002275**, paired UUID bootstrap 95% 구간은
**[−0.005480, +0.000660]**이다. 세 학습 seed 모두 JS 쪽 평균 NLL이 낮았지만
주 비교 구간에는 0이 포함된다. 이 실험만으로 JS가 Random보다 우수하다고
확정하지 않는다. JS − Fixed도 −0.001898, 95% 구간 [−0.005211, +0.001312]다.

| 질문 수 | Fixed NLL | Random NLL | Entropy NLL | JS NLL |
|---|---:|---:|---:|---:|
| 3 | 1.13204 | 1.13018 | 1.12925 | 1.12883 |
| 5 | 1.12519 | 1.12688 | 1.12438 | 1.12452 |
| 7 | 1.12269 | 1.12348 | 1.12340 | 1.12144 |
| 9 | 1.12321 | 1.12410 | 1.12244 | 1.12076 |

9개 예산만의 JS − Random 구간은 [−0.007316, −0.000046]이지만, 이는 여러
예산별 부차 비교 중 하나이며 다중 비교 보정 전 구간이다. 사전에 정한
예산 평균의 주 비교를 대신하지 않는다.

선택된 checkpoint와 학습 종료 시점을 구분해 기록했다.

| 학습 seed | 검증으로 선택한 epochs, 모델 5개 | 선택된 updates | 종료 시 updates |
|---|---|---|---|
| 2026 | 19 / 21 / 23 / 19 / 23 | 304 / 336 / 368 / 304 / 368 | 784 / 816 / 848 / 784 / 848 |
| 2027 | 24 / 20 / 33 / 23 / 22 | 384 / 320 / 528 / 368 / 352 | 864 / 800 / 1008 / 848 / 832 |
| 2028 | 29 / 23 / 25 / 27 / 28 | 464 / 368 / 400 / 432 / 448 | 944 / 848 / 880 / 912 / 928 |

학습 3회와 전체 정책 평가 3회가 모두 완료됐다. 각 seed의 검증 bank에는
60명과 180개 UUID/Context 조합이 모두 포함됐다. checkpoint나 학습 상한을
새 test 점수에 따라 바꾸지 않았다. 이전 60명 결과와 이번 300명 결과는
평가 표본과 응답 realization이 달라 직접적인 학습 효과 비교로 사용하지 않는다.

## 답변 이력과 문항 선택의 동작

같은 동결 모델, 같은 test 응답자, 같은 holdout 4문항에서 빈 이력 H0와
실제로 JS가 수집한 9개 응답 이력 H9를 비교했다. H0만 새로 예측하고 H9는
원래 rollout에 저장한 확률을 사용했다.

| 같은 모델에 제공한 이력 | NLL, seed 평균 ± SD | 정확도 |
|---|---:|---:|
| 빈 이력 H0 | 1.17134 ± 0.00286 | 45.657% |
| 실제 9개 응답 H9 | 1.12076 ± 0.00116 | 46.343% |

H9 − H0 NLL은 **−0.050584**, UUID paired 95% 구간 **[−0.078295, −0.026942]**다.
예측 분포의 평균 total variation은 0.03574, 예측 최빈 대안이 달라진 비율은
4.546%였다. 이 조건에서는 이력을 거의 무시하던 상태가 개선됐음을 확인했다.
이는 같은 모델의 이력 사용에 관한 진단이며 JS의 문항 선택 우위를 뜻하지 않는다.

문항 선택이 답변에 반응하는지는 초기 두 응답에 개입해 확인했다. 900개
UUID/Context episode에서 hash로 180개를 정하고, 같은 표본을 세 seed에
사용했다. 프로필·Context·초기 문항·남은 후보·모델·정책 난수는 그대로 두고
이미 공개된 두 답변만 `(choice + 1) % 4`로 바꿨다. 실제 세 번째 문항은
모든 진단 사례에서 원래 trace와 일치하게 재현됐다.

| 학습 seed | 답변을 바꾼 뒤 세 번째 문항이 달라진 사례 |
|---|---:|
| 2026 | 98 / 180 (54.44%) |
| 2027 | 93 / 180 (51.67%) |
| 2028 | 91 / 180 (50.56%) |

seed 평균은 **52.22%**다. 이는 같은 180개 사례를 세 모델로 진단한 값이며
독립 사례 540개를 뜻하지 않는다. 계산 과정의 답변 반응성을 보여주지만,
개입한 응답이 실제로 발생했을 때의 정책 성능이나 실제 정보이득을 추정하지는 않는다.

실제 전체 rollout에서도 JS와 Random의 9문항 순서는 세 seed 모두 900/900개
episode에서 달랐다. 최종 9문항의 집합도 각각 897/900, 899/900, 897/900개에서
달랐다. 초기 두 문항은 공통이며 세 번째부터 정책이 선택한다. JS 획득 점수의
중앙값은 seed별 약 0.00126~0.00152로, 거의 수치적 0만 반환하는 상태가 아니다.

## 실행 검수

독립 검수에서 세 seed의 저장된 holdout 예측을 실제 정답으로 다시 채점해
NLL·정확도·Brier와 집계 결과를 확인했다. 모든 bootstrap은 UUID를 묶었다.

첫 실행의 seed 2027 문항 선택 파일에서 마지막 12행이 빠진 상태를 발견했다.
당시 예측·점수 파일은 완전했다. 해당 파일을 보존하고 같은 checkpoint·입력·
프로토콜·난수·thread 설정으로 평가만 다시 실행했다. 새 trace는 32,400행으로
완전하며, 시간 측정값을 제외한 선택·응답·후보·점수 기준으로 기존 32,388행은
새 trace의 동일한 prefix였다. 모든 예측확률·질문
경로·NLL/정확도/Brier·기존 획득 점수의 차이는 0이었다. 모델이나 정책을
다시 선택하지 않았다. 원래 파일 상태의 원인은 확정하지 않았다.

## 재현

별도 전달된 원본 파일과 신뢰할 수 있는 Generator handoff가 필요하다. 원본
분할 manifest를 보관하고, 매번 새로운 출력 경로를 사용한다.

```bash
python scripts/prepare_recheck.py --source-data /local/original-inputs --original-split outputs/real-split.json --handoff /local/Response_Generator_Handoff --output-dir outputs/recheck --test-size 300 --seed 2026
python scripts/run_recheck.py --data outputs/recheck/data --split outputs/recheck/split.json --output-root outputs/recheck/runs --seeds 2026 2027 2028 --jobs 3
python scripts/summarize_recheck.py --evaluations outputs/recheck/runs/seed-2026/evaluation outputs/recheck/runs/seed-2027/evaluation outputs/recheck/runs/seed-2028/evaluation --seed-labels 2026 2027 2028 --output outputs/recheck/aggregate.json
python scripts/diagnose_recheck.py --data outputs/recheck/data --split outputs/recheck/split.json --runs outputs/recheck/runs/seed-2026 outputs/recheck/runs/seed-2027 outputs/recheck/runs/seed-2028 --output outputs/recheck/history-diagnostic.json --check-answer-sensitivity
```

`run_recheck.py`는 실행 명령과 설정을 먼저 저장하고, 학습 seed마다 별도
프로세스로 학습 → 평가를 실행한다. 각 프로세스는 CPU thread 1개를 사용한다.
실패한 부분 산출물을 재사용하거나 기존 결과를 덮어쓰지 않는다.

## 해석 범위

새 응답자에 대한 같은 문항은행·같은 Generator의 합성 응답 평가다. 새 문항,
실제 인간 응답, 다른 Generator, 다른 response seed에 대한 검증은 포함하지
않는다. train 표본은 180명이며, 이것이 충분한지 별도 규모 비교는 하지 않았다.
현재 JS는 예측 의존성의 proxy이며 정확한 Bayesian 정보이득은 아니다.
이력은 임의 부분집합으로 학습하고 정책 rollout 이력 보강은 사용하지 않았다.

원본 응답·프로필·Generator 코드와 계수·학습 checkpoint는 공개하지 않는다.
