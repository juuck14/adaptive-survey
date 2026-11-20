"""Validate one or more runs in one response CSV, optionally export logit long.

Uses Python standard library only. Does not fit models or reveal new responses.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import csv
import json
import math

HERE=Path(__file__).resolve().parent
PERSON=['GEN_Z','GEND_F','GEND_O','GEND_PNO','EDU_1','EDU_2','EDU_3','EDU_5','EDU_6']
ATTR=['DNS_B','DNS_A','MSA_P','MSA_F','WIFI_B','WIFI_H','PERF_A','PERF_F','ENG_B','ENG_F','ENV_2','ENV_3','CTSA_A']
LONG_EXTRA=['choice_set_id','alternative','alternative_order','availability','CHOICE']+ATTR
BASE={'transfer':1,'layover':3,'stopover':5}
GEN={'mnl','panel_mmnl'}
METHOD={'fixed','bayesian','non_graph','graph'}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        if len(set(fields))!=len(fields):raise ValueError('헤더에 중복된 열 이름이 있습니다.')
        rows=list(reader)
    return fields,rows

def require(ok,message):
    if not ok:raise ValueError(message)

def integer(value,label):
    try:return int(value)
    except (ValueError,TypeError):raise ValueError(f'{label}: 정수 값이 필요합니다 ({value!r}).')

def validate(path,expect_respondents=None,expect_per_context=None):
    manifest=json.loads((HERE/'package_manifest.json').read_text(encoding='utf8'))
    fields,rows=read_csv(path)
    required=manifest['response_columns']
    require(set(required)<=set(fields),'필수 열 누락: '+', '.join(sorted(set(required)-set(fields))))
    require(bool(rows),'응답 행이 없습니다. 헤더 전용 템플릿은 완료 자료가 아닙니다.')
    _,cat=read_csv(HERE/'task_catalog.csv')
    catalog=defaultdict(list)
    for r in cat:catalog[r['task_id']].append(r)
    require(len(cat)==216 and len(catalog)==54,'고정 문항 카탈로그 크기가 다릅니다.')
    for tid,alts in catalog.items():
        require([a['alternative'] for a in alts]==['A','B','C','NONE'],f'{tid}: 대안 순서가 잘못되었습니다.')
    _,initial=read_csv(HERE/'initial_tasks_main.csv')
    _,fixed=read_csv(HERE/'fixed_task_order.csv')
    initial_by_context={c:[r['task_id'] for r in initial if r['context']==c] for c in BASE}
    fixed_by_context={c:[r['task_id'] for r in fixed if r['context']==c] for c in BASE}
    strata={r['stratum_id']:r for r in manifest['profile_strata']}
    groups=defaultdict(list)
    public_profile={};public_response={}
    for line,r in enumerate(rows,2):
        where=f'{line}행'
        require(None not in r,f'{where}: 열 개수가 헤더보다 많습니다.')
        for f in required:
            if f not in ['acquisition_score','selection_time_ms']:
                require(r.get(f) not in [None,''],f'{where}: {f} 필수값 누락.')
        rep=integer(r['replicate_id'],where+' replicate_id')
        require(1<=rep<=10,f'{where}: 회차는 1~10입니다.')
        require(r['schema_version']==manifest['package_version'],f'{where}: schema_version 불일치.')
        require(r['generator_type'] in GEN and r['selection_method'] in METHOD,f'{where}: 생성환경/설문방식 값 오류.')
        expected_run=f"rep{rep:02d}_{r['generator_type']}_{r['selection_method']}"
        require(r['run_id']==expected_run,f'{where}: run_id는 {expected_run}이어야 합니다.')
        require(r['survey_instance_id']==expected_run+'__'+r['sample_id'],f'{where}: survey_instance_id 오류.')
        require(integer(r['sampling_seed'],where+' sampling_seed')==20261110+rep,f'{where}: 원래 회차별 sampling_seed와 다릅니다.')
        c=r['context'];require(c in BASE,f'{where}: context 오류.')
        require(integer(r['base_block'],where+' base_block')==BASE[c],f'{where}: 기준 블록 오류.')
        require(r['task_id'] in catalog,f'{where}: 문항 카탈로그에 없는 task_id.')
        t=catalog[r['task_id']][0]
        require(c==t['context'],f'{where}: 다른 Context의 문항이 포함되었습니다.')
        for f in ['task_block','question_in_block']:
            require(integer(r[f],where+' '+f)==integer(t[f],f),f'{where}: {f}가 문항 목록과 다릅니다.')
        order=integer(r['question_order'],where+' question_order')
        require(1<=order<=9,f'{where}: 제시 순서는 1~9입니다.')
        require(r['phase']==('initial' if order<=3 else 'subsequent'),f'{where}: phase와 순서 불일치.')
        require(r['chosen_alternative'] in ['A','B','C','NONE'],f'{where}: 선택 대안 오류. NONE도 문자열로 보존합니다.')
        require(r['stratum_id'] in strata,f'{where}: stratum_id 오류.')
        s=strata[r['stratum_id']]
        for f in PERSON:
            v=integer(r[f],where+' '+f)
            require(v in [0,1],f'{where}: {f}는 0/1이어야 합니다.')
            require(v==integer(s[f],f),f'{where}: {f}와 stratum_id 코딩 불일치.')
        for f in ['generation','gender','education']:
            require(r[f]==s[f],f'{where}: {f}와 stratum_id 불일치.')
        for f in ['acquisition_score','selection_time_ms']:
            if r[f] not in ['',None]:
                try:v=float(r[f])
                except ValueError:raise ValueError(f'{where}: {f}는 숫자 또는 빈칸이어야 합니다.')
                require(math.isfinite(v),f'{where}: {f}가 유한한 수가 아닙니다.')
                if f=='selection_time_ms':require(v>=0,f'{where}: 선택 시간은 음수일 수 없습니다.')
        pkey=(rep,r['sample_id'])
        pvalue=tuple(r[f] for f in ['uuid','context','stratum_id','generation','gender','education']+PERSON)
        require(pkey not in public_profile or public_profile[pkey]==pvalue,f'{where}: 같은 표본의 프로필/Context가 방법 간 달라졌습니다.')
        public_profile[pkey]=pvalue
        # Check matching potential answers only when both methods are present in the supplied CSV.
        akey=(rep,r['generator_type'],r['uuid'],r['context'],r['task_id'])
        require(akey not in public_response or public_response[akey]==r['chosen_alternative'],f'{where}: 같은 사람/문항의 답이 방법 간 다릅니다.')
        public_response[akey]=r['chosen_alternative']
        groups[r['survey_instance_id']].append(r)
    run_people=defaultdict(list)
    for instance,records in groups.items():
        records.sort(key=lambda r:int(r['question_order']))
        require(len(records)==9,f'{instance}: 응답은 9행이어야 합니다 ({len(records)}행).')
        require([int(r['question_order']) for r in records]==list(range(1,10)),f'{instance}: 순서 중복/누락.')
        require(len({r['task_id'] for r in records})==9,f'{instance}: 문항 중복.')
        immutable=[f for f in required if f not in ['task_id','task_block','question_in_block','question_order','phase','chosen_alternative','acquisition_score','selection_time_ms']]
        for f in immutable:require(len({r[f] for r in records})==1,f'{instance}: 개인의 {f}가 문항 간 변합니다.')
        c=records[0]['context']
        require([r['task_id'] for r in records[:3]]==initial_by_context[c],f'{instance}: 공통 초기 문항/순서 오류.')
        if records[0]['selection_method']=='fixed':
            require([r['task_id'] for r in records]==fixed_by_context[c],f'{instance}: Fixed 사전 순서 오류.')
        run_people[records[0]['run_id']].append(records[0])
    run_summary=[]
    for run,people in sorted(run_people.items()):
        require(len({r['uuid'] for r in people})==len(people),f'{run}: 회차 내 동일 UUID 중복.')
        counts=Counter(r['context'] for r in people)
        if expect_respondents is not None:require(len(people)==expect_respondents,f'{run}: 계획 인원 {expect_respondents}명과 다릅니다 ({len(people)}명).')
        if expect_per_context is not None:
            for c in BASE:require(counts[c]==expect_per_context,f'{run}: {c} 인원이 {expect_per_context}명이 아닙니다.')
            if expect_per_context==500:
                for c in BASE:
                    actual=Counter(r['stratum_id'] for r in people if r['context']==c)
                    for sid,s in strata.items():require(actual[sid]==int(s['quota_per_context']),f'{run}: {c}의 {sid} 층별 인원이 기존 quota와 다릅니다.')
        run_summary.append(dict(run_id=run,respondents=len(people),response_rows=len(people)*9,context_counts=dict(counts)))
    return fields,rows,catalog,run_summary

def convert(fields,rows,catalog,destination):
    require(not set(LONG_EXTRA)&set(fields),'응답 파일에 long-format 전용 열이 이미 포함되어 있습니다.')
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+LONG_EXTRA);writer.writeheader()
        for r in rows:
            for a in catalog[r['task_id']]:
                out=dict(r);out.update({k:a[k] for k in ['alternative','alternative_order','availability']+ATTR})
                out['choice_set_id']=r['survey_instance_id']+f"__q{int(r['question_order']):02d}"
                out['CHOICE']=int(r['chosen_alternative']==a['alternative'])
                writer.writerow(out)
    return len(rows)*4

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('responses');parser.add_argument('--output')
    parser.add_argument('--expect-respondents',type=int)
    parser.add_argument('--expect-per-context',type=int)
    args=parser.parse_args()
    try:
        if args.output:require(Path(args.output).resolve()!=Path(args.responses).resolve(),'원래 응답 파일과 변환 출력 경로를 다르게 지정하세요.')
        fields,rows,catalog,summary=validate(args.responses,args.expect_respondents,args.expect_per_context)
        result={'valid':True,'runs':summary,'response_rows':len(rows)}
        if args.output:result.update(long_rows=convert(fields,rows,catalog,args.output),output=str(Path(args.output).resolve()))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,FileNotFoundError) as e:
        parser.exit(1,f'검증 실패: {e}\n')

if __name__=='__main__':main()
