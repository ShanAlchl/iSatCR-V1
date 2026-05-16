from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from online_self_healing import BlockchainConsensusStateMachine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the output-driven blockchain consensus diagnosis demo."
    )
    parser.add_argument("--output-root-dir", default=None)
    parser.add_argument("--output-constellation-index", type=int, default=0)
    parser.add_argument("--round-index", type=int, default=25)
    parser.add_argument("--test-id", type=int, default=401)
    parser.add_argument("--event-id", default="consensus-demo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    engine = BlockchainConsensusStateMachine(output_root_dir=args.output_root_dir)
    report = engine.process_output_selection(
        {
            "output_constellation_index": args.output_constellation_index,
            "round_index": args.round_index,
            "test_id": args.test_id,
        }
    )
    payload = report.as_framework_event_payload(event_id=args.event_id)

    print(f"scenario_id={report.scenario_id}")
    print(f"tle_path={report.constellation_tle_path}")
    print(f"linked_failure_ratio={report.linked_failure_ratio:.4f}")
    print("result_diag:")
    for item in report.result_diag:
        print(
            "  SID={sid} HealthScore={health:.4f} RiskLevel={risk}".format(
                sid=item.SID,
                health=item.HealthScore,
                risk=item.RiskLevel,
            )
        )
    print("framework_payload:")
    print(
        json.dumps(
            {
                "EventId": payload["EventId"],
                "IsolationList": payload["IsolationList"],
                "ResultDiag": [
                    {
                        "SID": item.SID,
                        "HealthScore": item.HealthScore,
                        "RiskLevel": item.RiskLevel,
                    }
                    for item in payload["ResultDiag"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    # 保存结果到 result/consensus 子文件夹
    output_dir = REPO_ROOT / "result" / "consensus"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"report_{report.scenario_id}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        # 确保 ResultDiag 中的 NodeElement 对象被转换为字典以便 JSON 序列化
        serializable_payload = payload.copy()
        serializable_payload["ResultDiag"] = [
            item.as_dict() if hasattr(item, "as_dict") else item 
            for item in payload["ResultDiag"]
        ]
        
        json.dump({
            "scenario_id": report.scenario_id,
            "linked_failure_ratio": report.linked_failure_ratio,
            "framework_payload": serializable_payload
        }, f, ensure_ascii=False, indent=2)
    
    print(f"共识报告已保存至: {report_file}")
    
    print(
        "output_model_dir={model_dir}".format(
            model_dir=engine.resolve_output_model_dir(
                {
                    "output_constellation_index": args.output_constellation_index,
                    "round_index": args.round_index,
                    "test_id": args.test_id,
                }
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
