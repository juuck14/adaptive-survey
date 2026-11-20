# 공통 조건 대조 — 2026-11-20

11월 12일 전달본은 최종 설문 수집 규격이다. 11월 6일 구현 설명과 11월 9일
발표자료의 이전 질문 예산·holdout 설정은 그 규격을 덮어쓰지 않는다.

## 확인 결과

| 항목 | 근거 | 판정 |
|---|---|---|
| 초기 3문항, 총 9문항, Context별 후보 18개 | 전달본 2–3장과 지정 CSV | 완료한 실행에 반영 |
| 같은 표본·Context·저장 응답 | 전달본 1·4장 | 원본과 40개 산출물을 대조해 확인 |
| Graph acquisition 이름 | 11월 9일 발표자료 13장: `Acquisition Rule: Weighted JS` | 문서상 선택 기준 일치 |
| Weighted JS 정의 | 11월 6일 공통 구현 문서의 다른 후보 평균 weighted JS | 현재 `acquisition.py`와 식·가중치·대상 범위 일치 |
| Non-graph 입력·이력·가중치 갱신 | 발표자료 10장과 공통 구현 문서 | 프로필·문항 속성·mean-pooled 이력, 동결 가중치로 구현 |
| 열 회차 평가 UUID 제외 | 표본 manifest와 reserved UUID 목록 | 14,407명 합집합을 학습·validation에서 제외 |
| 공통 train/validation UUID | 공용 문서는 수민님 준비 예정으로 기록 | 최신 공통 분할 파일을 확인하지 못함 |
| 실제 GNN 구현의 feature/loss/분할/마스킹 | Graph 코드나 실행 설정이 필요 | 문서상 원칙은 확인, 코드 수준 대조는 미완료 |

완료한 Non-graph 실행은 별도로 선정한 학습 1,200명·validation 300명을 사용한다.
이 인원수가 11월 12일 규격을 위반한다는 근거는 확인되지 않았다. 팀의 실제 공통 분할과
일치한다는 주장도 하지 않는다. 공통 분할 파일이 확인된 뒤 UUID를 대조하고, 실제로
조건이 다를 때 실행 수정 여부를 결정한다. 현재 자료만으로 재실행이 필요하다고 결론내리지 않는다.

자료 확인 범위: 프로젝트 Data 공유 폴더와 For Adaptive Survey/Synthetic Respondents,
현재 공용 문서의 실험·Survey 탭, 11월 9일 발표자료, 11월 6일 공통 구현 설명,
11월 18일 저장된 단체 대화. 이 범위에서 최신 GNN 코드나 공통 분할 파일을 찾지 못했다.

## 완료 경계

현재 완료한 것은 Fixed와 Non-graph의 지정 설문 수집·파일 검증이다.
최종 생성 계수 회복 추정은 전달본 7장에 따라 Junyuk의 공통 추정기로 진행한다.
동일 조건의 Non-graph/GNN 성능 비교가 검증됐다고 보고하지 않는다.

공유 업로드는 승인 검토에서 차단되어 보류 중이다. 코드 변경은 PR #1에 있으며
이 문서 작성 시점에는 main에 병합하지 않았다. Overleaf 원고 반영은 로그인 때문에 미완료다.

## 자료

- [프로젝트 공용 문서](https://docs.google.com/document/d/1ZV1PQ-watURbW9PZrfmbb0D8aNIS3syqUwEg6SMHU1I/edit)
- [11월 9일 발표자료](https://docs.google.com/presentation/d/1zYMla6gqJJNEmbqx3TRXJMBYHR4Ip9AmrcHagx7PHag/edit)
- [11월 6일 공통 구현 설명](https://docs.google.com/document/d/1IJbD36AZ2E__IeOf_uxYjFh5BXUGSN97XxsymzMqFRc/edit)
