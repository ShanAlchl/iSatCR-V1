from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from online_self_healing import ConstellationFramework, SimulationOutput  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a one-click hybrid constellation simulation demo."
    )
    parser.add_argument("--agent-config-path", default=None)
    parser.add_argument("--tle-filepath", default=None)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--build-q-networks", action="store_true")
    parser.add_argument(
        "--q-network-init-mode",
        choices=("random", "file"),
        default="random",
    )
    parser.add_argument("--q-network-weight-dir", default=None)
    parser.add_argument("--output-root-dir", default=None)
    parser.add_argument("--output-constellation-index", type=int, default=None)
    parser.add_argument("--output-round-index", type=int, default=None)
    parser.add_argument("--output-test-id", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--event-offset-steps", type=int, default=1)
    parser.add_argument("--post-event-steps", type=int, default=2)
    parser.add_argument("--post-latch-steps", type=int, default=1)
    parser.add_argument("--event-satellite", default="Satellite_1100_1_1")
    parser.add_argument("--health-score", type=float, default=0.12)
    parser.add_argument("--risk-level", default="manual_isolation")
    parser.add_argument("--sdt", default="sdt-001")
    parser.add_argument("--event-id", default="evt-1")
    parser.add_argument("--heal-event-id", default="heal-now")
    return parser


def print_banner(title: str) -> None:
    print()
    print(f"=== {title} ===")


def print_outputs(label: str, outputs: Sequence[SimulationOutput]) -> None:
    print_banner(label)
    if not outputs:
        print("No SimulationOutput emitted in this phase.")
        return

    for output in outputs:
        graph = output.RawGraph
        isolated_nodes = list(graph.graph.get("isolated_nodes", ()))
        dynamic_state_time = graph.graph.get("DynamicStateTimeUTC", "<unknown>")
        event_ids = [
            event.EventId or f"anonymous@{event.EventTime.isoformat()}"
            for event in output.TriggeredEvents
        ]
        print(
            "step={step} time={time} dynamic_state_time={dynamic_time} reason={reason} "
            "isolation_flag={flag} state_changed={changed}".format(
                step=output.StepIndex,
                time=output.CurrentTime.isoformat(),
                dynamic_time=dynamic_state_time,
                reason=output.TriggerReason,
                flag=output.IsolationFlag,
                changed=output.StateChanged,
            )
        )
        print(
            "  nodes={nodes} edges={edges} isolated_nodes={isolated} "
            "active_latch={active} pending_events={pending}".format(
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                isolated=isolated_nodes,
                active=list(output.ActiveIsolationList),
                pending=output.PendingEventCount,
            )
        )
        print(
            "  triggered_events={events}".format(
                events=event_ids if event_ids else ["<none>"]
            )
        )


def print_framework_summary(framework: ConstellationFramework) -> None:
    print_banner("Framework Summary")
    print(f"agent_config_path={framework.agent_config_path}")
    print(f"tle_filepath={framework.tle_filepath}")
    print(f"start_time={framework.start_time.isoformat()}")
    print(f"time_step_seconds={framework.time_step_seconds}")
    print(f"event_wait_step_seconds={framework.event_wait_step_seconds}")
    print(f"duration_steps={framework.duration_steps}")
    print(f"build_q_networks={framework.build_q_networks}")
    print(f"q_network_init_mode={framework.q_network_init_mode}")
    print(f"q_network_weight_dir={framework.q_network_weight_dir}")
    print(f"resolved_q_network_weight_dir={framework.resolved_q_network_weight_dir}")
    print(f"output_root_dir={framework.output_root_dir}")
    print(f"output_constellation_index={framework.output_constellation_index}")
    print(f"output_round_index={framework.output_round_index}")
    print(f"output_test_id={framework.output_test_id}")
    print(
        "yaml_begin_time={begin} yaml_time_stride={stride} yaml_duration={duration} yaml_rounds={rounds}".format(
            begin=framework.runtime_config.begin_time.isoformat(),
            stride=framework.runtime_config.coarse_time_stride_seconds,
            duration=framework.runtime_config.duration_intervals,
            rounds=framework.runtime_config.rounds,
        )
    )


def build_framework(args: argparse.Namespace) -> ConstellationFramework:
    framework_kwargs = {
        "emit_initial_state": True,
        "emit_on_topology_change": True,
    }
    if args.agent_config_path:
        framework_kwargs["agent_config_path"] = args.agent_config_path
    if args.tle_filepath:
        framework_kwargs["tle_filepath"] = args.tle_filepath
    if args.start_time:
        framework_kwargs["start_time"] = args.start_time
    if args.build_q_networks:
        framework_kwargs["build_q_networks"] = True
    framework_kwargs["q_network_init_mode"] = args.q_network_init_mode
    if args.q_network_weight_dir:
        framework_kwargs["q_network_weight_dir"] = args.q_network_weight_dir
    if args.output_root_dir:
        framework_kwargs["output_root_dir"] = args.output_root_dir
    if args.output_constellation_index is not None:
        framework_kwargs["output_constellation_index"] = args.output_constellation_index
    if args.output_round_index is not None:
        framework_kwargs["output_round_index"] = args.output_round_index
    if args.output_test_id is not None:
        framework_kwargs["output_test_id"] = args.output_test_id
    return ConstellationFramework(**framework_kwargs)


def run_demo(args: argparse.Namespace) -> int:
    framework = build_framework(args)
    print_framework_summary(framework)

    warmup_outputs = framework.run_steps(args.warmup_steps)
    print_outputs("Warmup", warmup_outputs)

    event_time = framework.current_time + (framework.time_step * args.event_offset_steps)
    framework.inject_event(
        event_time=event_time,
        ResultDiag=[
            (
                args.event_satellite,
                args.health_score,
                args.risk_level,
                args.sdt,
            )
        ],
        IsolationList=[args.event_satellite],
        EventId=args.event_id,
    )
    print_banner("Injected Isolation Event")
    print(
        "event_id={event_id} event_time={event_time} target={target}".format(
            event_id=args.event_id,
            event_time=event_time.isoformat(),
            target=args.event_satellite,
        )
    )

    event_outputs = framework.run_steps(
        args.event_offset_steps + args.post_event_steps
    )
    print_outputs("Event Processing", event_outputs)

    post_latch_outputs = framework.run_steps(args.post_latch_steps)
    print_outputs("Post-Latch Drift", post_latch_outputs)
    print(
        "latched_isolation_after_drift={latched}".format(
            latched=list(framework.current_snapshot.RawGraph.graph.get("isolated_nodes", ()))
            if framework.current_snapshot is not None
            else []
        )
    )

    framework.inject_event(HealFlag=True, EventId=args.heal_event_id)
    print_banner("Injected Heal Event")
    print(
        "event_id={event_id} event_time={event_time}".format(
            event_id=args.heal_event_id,
            event_time=framework.current_time.isoformat(),
        )
    )
    heal_output = framework.flush_events()
    heal_outputs: Iterable[SimulationOutput] = [] if heal_output is None else [heal_output]
    print_outputs("Heal Processing", list(heal_outputs))

    print_banner("Final State")
    print(f"current_time={framework.current_time.isoformat()}")
    print(f"step_index={framework.step_index}")
    print(f"active_isolation_list={list(framework.active_isolation_list)}")
    print(f"pending_event_count={framework.pending_event_count}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
