import base64
import hashlib
import json
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent; root=here.parents[2]
package=root/'collaborative_test/SW_0184_train_readout_alignment'
rows=json.loads((here/'completed_packet.json').read_text(encoding='utf-8-sig'))
reports={}; states={}
for row in rows:
    raw=base64.b64decode(row['data']); assert hashlib.sha256(raw).hexdigest()==row['sha256']
    target=package/'results_archive/result_v1'/row['name']; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
    obj=json.loads(raw); seed=int(row['name'][4])
    if row['name'].endswith('_report.json'):reports[seed]=(obj,row['sha256'])
    else:states[seed]=obj
assert set(reports)==set(states)=={0,1,2}
summary={}
for seed,(report,digest) in reports.items():
    assert states[seed]['status']=='complete' and states[seed]['report_sha256']==digest
    assert report['optimizer_updates']==0 and report['validation_data_read'] is False
    assert set(report['seeds'])=={str(seed)}
    assert len(report['cohort_ids'])==len(set(report['cohort_ids']))==16
    assert report['positions']==list(range(0,1024,64))
    for path,sha in report['implementation_fingerprint']['files'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha
    s=report['seeds'][str(seed)]; summary[str(seed)]={}
    for arm,v in s['arms'].items():
        model=s['models'][arm]
        assert model['before_sha256']==model['after_sha256'] and model['state_unchanged'] is True
        assert all(len(v[key])==16 and np.isfinite(v[key]).all() for key in ('qce','margin'))
        bg=[]
        for item in v['pairs']:
            a,f=item['all'],item['foreground']
            assert a['valid_classes']==f['valid_classes']+1
            value=a['same_threshold_recall']*a['valid_classes']-f['same_threshold_recall']*f['valid_classes']
            assert -1e-12<=value<=1+1e-12
            bg.append(value)
        summary[str(seed)][arm]=dict(qce=float(np.mean(v['qce'])),old8_qce=float(np.mean(v['qce'][:8])),
            new8_qce=float(np.mean(v['qce'][8:])),margin=float(np.mean(v['margin'])),
            foreground_same_threshold_recall=float(np.mean([x['foreground']['same_threshold_recall'] for x in v['pairs']])),
            background_same_threshold_recall=float(np.mean(bg)),
            metrics={ep:{metric:float(np.mean(values)) for metric,values in metrics.items()}
                for ep,metrics in v['metrics'].items()})
audit=dict(status='complete_TRAIN16_report_provenance_arithmetic_and_model_immutability_verified',
    summary=summary,optimizer_updates=0,validation_data_read=False,incumbent_changed=False,
    independent_mask_inference_replayed=False,
    limitation='This audit verifies stored report provenance and arithmetic, not an independent second inference or metric implementation.')
(package/'results_archive/result_v1/summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(package/'report_audit_source.py').write_bytes(Path(__file__).read_bytes())
text='''# SW0184 완료: TRAIN16 손실/분할 진단

고정 TRAIN16(old8+new8), frozen183 두 모델을 비교했다. 추가 학습 없이
실제 S→native Q→QCC→SW165 분할을 세 seed에서 실행했다.

| Seed | QCE 512 → 1024 | Standalone FG-ARI 512 → 1024 | Foreground IoU 512 → 1024 |
|---|---:|---:|---:|
| 0 | .244963 → .294692 | .913664 → .895067 | .684570 → .819372 |
| 1 | .408745 → .361419 | .835867 → .837792 | .611928 → .598299 |
| 2 | .365583 → .369496 | .864376 → .873369 | .526855 → .488036 |

이 숫자는 validation 성능이 아니며 .87 목표 달성으로 세지 않는다.
seed0는 QCE/FG가 함께 악화되고 seed1은 함께 개선됐다. seed2는 QCE가
조금 높아졌지만 FG는 개선됐다. 따라서 모든 seed에 공통인 단일 원인을
확정하지 않는다. 기존183의 full320 결과 및 최고 모델도 바꾸지 않는다.

같은 BG patch 쌍의 Q>=.50 비율은 candidate seed0/1/2에서 각각
.8255/.6268/.3498이다. FG same-object 쌍은 .8536/.8081/.8442이다.
이는 특히 seed2의 배경 연결이 약한 현상과 일치하지만, 모든 쌍의 평균을
공간 graph의 connectivity나 오류의 인과 증명으로 취급하지 않는다.
BG 비율은 기록된 all-class/FG-class balanced 평균과 class 수로 역산했고
모든16행에서 정확히 BG 한 class 차이가 있음을 검사했다.

다음 모델 변경은 QCE를 유지하면서 TRAIN mask의 공간 이웃 same-class
연결과 cross-class 경계가 실제 QCC .50 기준 양쪽에 놓이도록 보조 loss를
주는 것이다. 양의 연결은 per-component signed rho에 직접 credit을 주고,
경계 억제는 production Q를 사용한다. 이전 forest-loss 단독 실패와 구분해
새 실험으로 등록하고, 실제 분할이 좋아질 때만 확대한다.

원본 보고서의 no-update model tensor hash, cohort/producer SHA와 산술은
검증했다. 이 사후 검사는 두 번째 mask 추론이나 독립 metric 재계산은
아니다. scope는16장에 한정한다.

원본과 집계: [result_v1](results_archive/result_v1/).
'''
(package/'RESULT_KO.md').write_text(text,encoding='utf-8')
(package/'README.md').write_text('# SW0184 — completed TRAIN-only alignment diagnostic\n\n'
    'All three seed replays completed with zero optimizer updates and unchanged model tensors.\n'
    'The evidence is mixed across seeds; no single universal cause is proven.\n'
    'See [results and next direction](RESULT_KO.md). No fixed320 score or incumbent changed.\n',encoding='utf-8')
path=root/'collaborative_test/best_model_tracking_20261008.json'
tracker=json.loads(path.read_text(encoding='utf-8'))
tracker['latest_train_readout_alignment_20261016'].update(status='complete_three_seed_TRAIN16_diagnostic',
    summary=summary,incumbent_changed=False,
    result_path='collaborative_test/SW_0184_train_readout_alignment/RESULT_KO.md')
with path.open('w',encoding='utf-8',newline='\n') as stream:
    json.dump(tracker,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
print('SW0184 completed; stored provenance/arithmetic checked; incumbent unchanged')
