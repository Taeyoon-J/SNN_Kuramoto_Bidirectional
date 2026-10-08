"""Validate and summarize the saved, GT-free SW0108 activation diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STREAM = REPO.parent / "collaborative_private_runtime_20261008/SW0108_four_diagnostics.jsonstream"
PARTIAL = HERE / "scientific_partial_20261008.json"
OUTPUT = HERE / "activation_summary_20261008.json"
NOTE = HERE / "research_note_ko_20261008.md"
ARMS = ("baseline", "kuramoto_K0", "constant_gate_half", "dendrite_no_retention")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
EXPECTED_CORE_SHA = "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf"
EXPECTED_REFERENCE_SHA = "009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_json_stream(text: str):
    decoder = json.JSONDecoder()
    position = 0
    rows = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        row, position = decoder.raw_decode(text, position)
        if not isinstance(row, dict):
            raise ValueError("diagnostic stream entries must be JSON objects")
        rows.append(row)
    if not rows:
        raise ValueError("diagnostic stream is empty")
    return rows


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite value at {label}")
    return value


def validate_diagnostics(rows):
    if tuple(row.get("arm") for row in rows) != ARMS:
        raise ValueError("diagnostic arm order/count differs from the four completed registered arms")
    provenance_keys = ("source_checkpoint_sha256", "screen_runner_sha256",
                       "shared_evaluator_sha256", "protocol_sha256")
    first = rows[0]
    for row in rows:
        if row.get("status") != "complete" or row.get("ground_truth_used") is not False:
            raise ValueError(f"{row.get('arm')} is incomplete or used ground truth")
        for key in provenance_keys:
            if row.get(key) != first.get(key):
                raise ValueError(f"diagnostic provenance mismatch in {key}")
        if row.get("source_checkpoint_sha256") != EXPECTED_CORE_SHA:
            raise ValueError("diagnostics do not use the registered SW0097 seed0 source core")
        if not str(row.get("source_checkpoint", "")).endswith(
                "SW0097_graph_adaptation/seed0_positive_frozen/core.pt"):
            raise ValueError("unexpected diagnostic checkpoint identity")
        trace = row["trace"]
        if trace.get("settle") != 512 or trace.get("steps") != 1024:
            raise ValueError(f"unexpected temporal window for {row['arm']}")
        if row["arm"] == "baseline":
            guard = row.get("registered_SW0097_baseline_match", {})
            if guard.get("passed") is not True or guard.get("reference_sha256") != EXPECTED_REFERENCE_SHA:
                raise ValueError("baseline diagnostic lacks a passing exact SW0097 reference guard")
        expected_counts = {
            "all_frames": {"gate": 83_886_080, "event_before_override": 335_544_320,
                           "spike_output": 335_544_320},
            "settled_frames": {"gate": 41_943_040, "event_before_override": 167_772_160,
                               "spike_output": 167_772_160},
        }
        for scope in ("all_frames", "settled_frames"):
            for signal in ("gate", "event_before_override", "spike_output"):
                stat = trace[scope][signal]
                if int(stat["count"]) != expected_counts[scope][signal]:
                    raise ValueError(f"unexpected observation count at {row['arm']}.{scope}.{signal}")
                for field in ("mean", "std", "min", "max"):
                    finite(stat[field], f"{row['arm']}.{scope}.{signal}.{field}")
        per_unit = trace["per_unit_settled_time"]
        for signal in ("gate", "event"):
            info = per_unit[signal]
            if int(info.get("batches", 0)) != 40:
                raise ValueError(f"{row['arm']} {signal} temporal stats do not cover all 40 batches")
            finite(info["temporal_std_mean"], f"{row['arm']}.{signal}.temporal_std_mean")
            finite(info["constant_unit_fraction_mean"], f"{row['arm']}.{signal}.constant_unit_fraction_mean")
        if not 0 <= per_unit["event"]["always_on_fraction_mean"] <= 1:
            raise ValueError(f"invalid always-on fraction for {row['arm']}")
    # These three expected file hashes tie the saved stream to the current local source.
    expected_files = {
        "screen_runner_sha256": HERE / "run.py",
        "shared_evaluator_sha256": REPO / "collaborative_test/SW_0040_peer_transfer/evaluate.py",
        "protocol_sha256": HERE / "protocol.json",
    }
    for key, path in expected_files.items():
        if sha(path) != first[key]:
            raise ValueError(f"saved diagnostics do not match current {path.name} ({key})")
    return first


def validate_partial(report):
    if (report.get("seed") != 0 or report.get("images") != 320
            or report.get("ids") != [1320, 1639]
            or report.get("status") != "partial_four_completed_two_pending"):
        raise ValueError("partial endpoint metadata differs from registered SW0108 scope")
    if tuple(report.get("metrics", {}).keys()) != ARMS:
        raise ValueError("partial metric arms/count differ from the four completed arms")
    if report.get("remaining_arms") != ["membrane_no_retention", "events_forced_on"]:
        raise ValueError("partial report does not preserve the two pending arms")
    if report.get("bootstrap", {}).get("unit") != "paired image":
        raise ValueError("expected paired-image bootstrap metadata")
    for arm in ARMS:
        for metric in METRICS:
            finite(report["metrics"][arm][metric], f"{arm}.{metric}")
    for arm in ARMS[1:]:
        for metric in METRICS:
            ci = report["comparisons_to_baseline"][arm][metric]["paired_image_ci95"]
            if len(ci) != 2 or not all(math.isfinite(float(x)) for x in ci) or ci[0] > ci[1]:
                raise ValueError(f"invalid paired CI for {arm}.{metric}")


def summarize(rows, partial, provenance):
    out = {
        "experiment": "SW0108_gate_contribution_screen",
        "status": "partial_four_of_six_arms_validated",
        "data_scope": {"seed": 0, "global_ids_inclusive": [1320, 1639], "images": 320,
                       "batch_size": 8, "steps": 1024, "settle": 512,
                       "prediction_gt_used": False},
        "provenance": {
            "source": "SW0097_graph_adaptation/seed0_positive_frozen/core.pt",
            "source_core_sha256": provenance["source_checkpoint_sha256"],
            "screen_runner_sha256": provenance["screen_runner_sha256"],
            "shared_evaluator_sha256": provenance["shared_evaluator_sha256"],
            "protocol_sha256": provenance["protocol_sha256"],
            "baseline_reference_sha256": EXPECTED_REFERENCE_SHA,
        },
        "activations": {},
        "endpoint_metrics": partial["metrics"],
        "paired_image_comparisons_to_baseline": partial["comparisons_to_baseline"],
        "bootstrap_scope": partial["bootstrap"],
        "remaining_arms": partial["remaining_arms"],
        "interpretation_limits": [
            "One frozen seed and one-at-a-time inference interventions; not a trained causal effect or data-scaling explanation.",
            "Scalar hooks do not provide per-object mask correlation.",
            "Constant gate changes both carrier amplitude/temporal modulation and membrane gate.",
            "K=0 collapse is evidence only in this frozen-parameter context.",
            "Dendrite-retention paired intervals include zero; no clear endpoint change is resolved.",
        ],
    }
    for row in rows:
        tr = row["trace"]
        settled = tr["settled_frames"]
        per_unit = tr["per_unit_settled_time"]
        out["activations"][row["arm"]] = {
            "settled_gate": {k: settled["gate"][k] for k in ("mean", "std")},
            "settled_event_before_intervention": {
                k: settled["event_before_override"][k] for k in ("mean", "std")},
            "settled_actual_spike_output": {k: settled["spike_output"][k] for k in ("mean", "std")},
            "per_unit_temporal": {
                "gate_temporal_std_mean": per_unit["gate"]["temporal_std_mean"],
                "gate_constant_unit_fraction_mean": per_unit["gate"]["constant_unit_fraction_mean"],
                "event_temporal_std_mean": per_unit["event"]["temporal_std_mean"],
                "event_constant_unit_fraction_mean": per_unit["event"]["constant_unit_fraction_mean"],
                "event_always_on_fraction_mean": per_unit["event"]["always_on_fraction_mean"],
            },
        }
    return out


def render_note(summary):
    baseline = summary["activations"]["baseline"]
    k0 = summary["paired_image_comparisons_to_baseline"]["kuramoto_K0"]
    half = summary["paired_image_comparisons_to_baseline"]["constant_gate_half"]
    dend = summary["paired_image_comparisons_to_baseline"]["dendrite_no_retention"]
    return f"""# SW0108: 게이트 기여도 스크리닝 (부분 결과)

이 분석은 SW0097 seed 0의 고정된 체크포인트에 등록된 단일 개입을 적용한 GT-free 추론 민감도 실험이다. 검증 이미지 320장(ID 1320–1639), batch 8, 1024 step 중 마지막 512 step을 분석했다. 여섯 개 arm 중 baseline과 세 개 개입만 완료됐고, `membrane_no_retention`과 `events_forced_on`은 미완료다.

구현 경로에서 막 전위 계층은 매 시점 `e = act_fun_adp(inputs_)`를 계산하고 실제 출력을 `spike = e * g_wave_t`로 만든다. 이벤트는 임계값 기반 이진 함수이고, gate는 펼쳐진 scalar `g_wave_t`다. 따라서 baseline의 마지막 절반 이벤트 평균 {baseline['settled_event_before_intervention']['mean']:.6f}은 이벤트가 거의 항상 열려 있음을 시사한다. settled gate 평균/표준편차는 {baseline['settled_gate']['mean']:.3f}/{baseline['settled_gate']['std']:.3f}; gate의 단위별 시간 표준편차 평균은 {baseline['per_unit_temporal']['gate_temporal_std_mean']:.3f}이다. 이벤트 단위의 {100*baseline['per_unit_temporal']['event_always_on_fraction_mean']:.2f}%는 마지막 절반 내내 항상 켜져 있었다. 이는 이벤트 선택기가 자주 열린다는 관측이며 막 동역학 전체가 중요하지 않다는 뜻은 아니다.

`kuramoto_K0`는 FG-ARI가 크게 하락했고 paired-image 95% CI는 [{k0['fg_ari']['paired_image_ci95'][0]:.3f}, {k0['fg_ari']['paired_image_ci95'][1]:.3f}]였다. 이는 이 고정된 파라미터 상태에서 결합을 제거한 추론이 성능을 무너뜨렸다는 증거다. `constant_gate_half`에서는 실제 spike가 거의 상수 0.5가 되었고 세 지표 모두 0이었다. 이 개입은 막 gate와 함께 carrier를 `0.5*sin(theta)`로 바꾸므로 gate의 시간 변조와 carrier 크기 효과를 분리하지 못한다. `dendrite_no_retention` 결과의 FG-ARI 변화 CI는 [{dend['fg_ari']['paired_image_ci95'][0]:.4f}, {dend['fg_ari']['paired_image_ci95'][1]:.4f}]로 0을 포함한다. 이 단일 seed 자료에서 뚜렷한 성능 변화를 확인하지 못했다.

이 결과는 학습된 인과효과, 전체 membrane 기여도의 부재, 데이터 규모에 대한 설명을 확정하지 않는다. scalar hook은 객체 mask와의 상관을 측정하지 않는다. 특히 bootstrap은 이미지 쌍을 재표집한 단일 학습 seed 조건부 구간이다. 남은 두 arm 완료 후 해석을 갱신할 수 있지만, 이 결과만으로 새 하이퍼파라미터를 선택하지 않는다.
"""


def main():
    rows = decode_json_stream(STREAM.read_text(encoding="utf-8"))
    provenance = validate_diagnostics(rows)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
    validate_partial(partial)
    summary = summarize(rows, partial, provenance)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    NOTE.write_text(render_note(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "objects": len(rows),
                      "summary": str(OUTPUT), "note": str(NOTE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
