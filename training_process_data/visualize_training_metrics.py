import argparse
import html
import json
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    suffix: str
    axis_title: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    aliases: tuple[str, ...] = ()


METRIC_SPECS: List[MetricSpec] = [
    MetricSpec("packet_loss_rate", "PacketLossRate", "%", "PacketLossRate (%)", min_value=0.0, max_value=100.0),
    MetricSpec("network_throughput", "NetworkThroughput", " Mbps", "NetworkThroughput (Mbps)", min_value=0.0),
    MetricSpec("bandwidth_utilization", "BandwidthUtilization", "%", "BandwidthUtilization (%)", min_value=0.0, max_value=100.0),
    MetricSpec("avg_packet_node_visits", "AvgPacketNodeVisits", "", "AvgPacketNodeVisits", min_value=0.0),
    MetricSpec("cumulative_reward", "CumulativeReward", "", "CumulativeReward"),
    MetricSpec("average_inference_time", "AverageInferenceTime", " ms", "AverageInferenceTime (ms)", min_value=0.0),
    MetricSpec(
        "average_e2e_delay",
        "AverageE2eDelay(Average delay for successful transmissions)",
        " seconds",
        "AverageE2eDelay (seconds)",
        min_value=0.0,
    ),
    MetricSpec(
        "average_hop_count",
        "AverageHopCount(Average hop count for successful transmissions)",
        " hops",
        "AverageHopCount (hops)",
        min_value=0.0,
    ),
    MetricSpec(
        "computing_ratio",
        "Proportion of satellites in computation",
        "%",
        "Satellites In Computation (%)",
        min_value=0.0,
        max_value=100.0,
        aliases=("AverageComputingRatio",),
    ),
    MetricSpec(
        "average_computing_waiting_time",
        "Average waiting time for computing",
        " seconds",
        "Average Waiting Time For Computing (seconds)",
        min_value=0.0,
        aliases=("ComputingWaitingTime",),
    ),
    MetricSpec(
        "average_ending_reward",
        "Average ending reward",
        "",
        "AverageEndingReward",
        aliases=("AverageEndingReward",),
    ),
]

METRIC_BY_KEY = {metric.key: metric for metric in METRIC_SPECS}
METRIC_BY_LABEL = {metric.label: metric for metric in METRIC_SPECS}
STEP_PATTERN = re.compile(r"^====== step (\d+) ======$")
TIME_PATTERN = re.compile(r"^====== (?!step )(.+?) ======$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize metric trends from a training metric log file."
    )
    parser.add_argument("input_file", type=str, help="Path to a training metric text file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metric keys to visualize. Use --list-metrics to see supported keys.",
    )
    parser.add_argument(
        "--run",
        default="all",
        help="Deprecated. The entire log file is treated as a single training run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Defaults to <input_stem>_metrics.html beside the log file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive figure after writing the HTML file.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List supported metrics and exit.",
    )
    return parser.parse_args()


def list_metrics():
    for metric in METRIC_SPECS:
        print(f"{metric.key:32} -> {metric.label}")


def _parse_metric_value(raw_value: str, suffix: str) -> Optional[float]:
    raw_value = raw_value.strip()
    if raw_value == "None":
        return None
    if suffix and raw_value.endswith(suffix):
        raw_value = raw_value[: -len(suffix)].strip()
    return float(raw_value)


def _parse_metric_line(line: str) -> Dict[str, float]:
    parsed = {}
    for metric in METRIC_SPECS:
        candidate_labels = (metric.label, *metric.aliases)
        for candidate_label in candidate_labels:
            prefix = f"{candidate_label}:"
            if not line.startswith(prefix):
                continue
            value = _parse_metric_value(line[len(prefix):], metric.suffix)
            if value is not None:
                parsed[metric.key] = value
            break
    return parsed


def parse_training_log(file_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    current_block: Optional[Dict[str, object]] = None

    def finalize_block():
        nonlocal current_block
        if current_block is None:
            return
        records.append(current_block)
        current_block = None

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            step_match = STEP_PATTERN.match(line)
            if step_match:
                finalize_block()
                current_block = {
                    "step": int(step_match.group(1)),
                    "display_time": None,
                    "metrics": {},
                }
                continue

            if current_block is None:
                continue

            time_match = TIME_PATTERN.match(line)
            if time_match:
                current_block["display_time"] = time_match.group(1)
                continue

            current_block["metrics"].update(_parse_metric_line(line))

    finalize_block()
    return records


def resolve_metrics(requested_metrics: List[str], records: List[Dict[str, object]]) -> List[MetricSpec]:
    available_metric_keys = set()
    for block in records:
        available_metric_keys.update(block["metrics"].keys())

    if not available_metric_keys:
        raise ValueError("No supported metrics were found in the input file.")

    if requested_metrics == ["all"]:
        return [metric for metric in METRIC_SPECS if metric.key in available_metric_keys]

    selected_specs = []
    for metric_key in requested_metrics:
        if metric_key not in METRIC_BY_KEY:
            raise ValueError(f"Unsupported metric key: {metric_key}")
        if metric_key not in available_metric_keys:
            raise ValueError(f"Metric '{metric_key}' was not found in the input file.")
        selected_specs.append(METRIC_BY_KEY[metric_key])
    return selected_specs


def _format_number(value: float) -> str:
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.01):
        return f"{value:.3e}"
    return f"{value:.3f}"


def _render_metric_svg(metric: MetricSpec, series_list: List[Dict[str, object]]) -> str:
    width = 1100
    height = 280
    left = 90
    right = 30
    top = 30
    bottom = 50
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_steps = [point["step"] for series in series_list for point in series["points"]]
    all_values = [point["value"] for series in series_list for point in series["points"]]
    if not all_steps or not all_values:
        raise ValueError(f"No data points were found for metric '{metric.key}'.")

    x_min = min(all_steps)
    x_max = max(all_steps)
    y_min = min(all_values)
    y_max = max(all_values)

    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        padding = abs(y_min) * 0.05 if y_min != 0 else 1.0
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    if metric.min_value is not None:
        y_min = max(y_min, metric.min_value)
    if metric.max_value is not None:
        y_max = min(y_max, metric.max_value)

    if y_min == y_max:
        if metric.max_value is not None and y_max < metric.max_value:
            y_max = min(metric.max_value, y_max + 1.0)
        elif metric.min_value is not None and y_min > metric.min_value:
            y_min = max(metric.min_value, y_min - 1.0)
        else:
            y_min -= 1.0
            y_max += 1.0

    def scale_x(step: float) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg_parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" class="metric-svg" role="img" '
            f'aria-label="{html.escape(metric.axis_title)}" '
            f'data-plot-left="{left}" data-plot-right="{width - right}" '
            f'data-plot-top="{top}" data-plot-bottom="{height - bottom}">'
        )
    ]
    svg_parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" rx="10" />')

    y_tick_count = 5
    for index in range(y_tick_count + 1):
        tick_value = y_min + (y_max - y_min) * index / y_tick_count
        y = scale_y(tick_value)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick-label">{html.escape(_format_number(tick_value))}</text>'
        )

    x_tick_count = min(6, len(set(all_steps))) or 1
    for index in range(x_tick_count + 1):
        tick_step = x_min + (x_max - x_min) * index / x_tick_count
        x = scale_x(tick_step)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#f3f4f6" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" class="tick-label">{int(round(tick_step))}</text>'
        )

    svg_parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="1.4" />'
    )
    svg_parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="1.4" />'
    )
    svg_parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" class="axis-label">Training Step</text>'
    )
    svg_parts.append(
        f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" class="axis-label" transform="rotate(-90 24 {height / 2:.2f})">{html.escape(metric.axis_title)}</text>'
    )

    for series in series_list:
        interactive_points = []
        points_attr = " ".join(
            f"{scale_x(point['step']):.2f},{scale_y(point['value']):.2f}" for point in series["points"]
        )
        for point in series["points"]:
            interactive_points.append(
                {
                    "x": round(scale_x(point["step"]), 2),
                    "y": round(scale_y(point["value"]), 2),
                    "step": int(point["step"]),
                    "value": float(point["value"]),
                    "display_value": _format_number(point["value"]),
                    "time": point["time"] or "",
                }
            )
        points_json = html.escape(json.dumps(interactive_points, ensure_ascii=False), quote=True)
        series_name = html.escape(str(series["name"]), quote=True)
        series_color = html.escape(str(series["color"]), quote=True)
        svg_parts.append("<g>")
        svg_parts.append(
            f'<polyline class="metric-line" fill="none" stroke="{series["color"]}" stroke-width="2.2" points="{points_attr}" />'
        )
        svg_parts.append(
            f'<polyline class="metric-hit-area" fill="none" stroke="rgba(15,23,42,0.001)" stroke-width="18" points="{points_attr}" '
            f'data-points="{points_json}" data-series-name="{series_name}" data-series-color="{series_color}" />'
        )
        svg_parts.append("</g>")

    svg_parts.append(
        f'<g class="hover-layer" visibility="hidden" pointer-events="none">'
        f'<line class="hover-guide" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />'
        f'<circle class="hover-anchor" cx="{left}" cy="{top}" r="5.5" />'
        f"</g>"
    )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def build_html_report(
    file_path: Path,
    records: List[Dict[str, object]],
    selected_metrics: List[MetricSpec],
    title: Optional[str],
) -> str:
    line_color = "#2563eb"

    metric_sections = []
    for metric in selected_metrics:
        points_by_step: Dict[int, Dict[str, object]] = {}
        for block in records:
            if metric.key not in block["metrics"]:
                continue
            step = int(block["step"])
            points_by_step[step] = {
                "step": step,
                "value": float(block["metrics"][metric.key]),
                "time": block.get("display_time") or "",
            }
        points = [points_by_step[step] for step in sorted(points_by_step)]

        series_list = []
        if points:
            series_list.append(
                {
                    "name": metric.axis_title,
                    "color": line_color,
                    "points": points,
                }
            )

        if not series_list:
            raise ValueError(f"No data points were found for metric '{metric.key}'.")

        metric_sections.append(
            "\n".join(
                [
                    '<section class="metric-card">',
                    f"<h2>{html.escape(metric.axis_title)}</h2>",
                    '<div class="chart-wrap">',
                    _render_metric_svg(metric, series_list),
                    f'<div class="chart-tooltip" hidden data-value-suffix="{html.escape(metric.suffix, quote=True)}"></div>',
                    "</div>",
                    "</section>",
                ]
            )
        )

    metric_summary = ", ".join(metric.key for metric in selected_metrics)
    page_title = title or f"Training Metric Trends: {file_path.name}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .summary {{
      margin: 0 0 24px;
      color: #475569;
      line-height: 1.6;
    }}
    .metric-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }}
    .metric-card h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 10px;
      color: #334155;
      font-size: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .metric-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-wrap {{
      position: relative;
    }}
    .metric-line {{
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .metric-hit-area {{
      cursor: crosshair;
      pointer-events: stroke;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .hover-guide {{
      stroke: #94a3b8;
      stroke-width: 1.2;
      stroke-dasharray: 4 4;
    }}
    .hover-anchor {{
      stroke: #ffffff;
      stroke-width: 2;
    }}
    .chart-tooltip {{
      position: absolute;
      min-width: 132px;
      max-width: 240px;
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.94);
      color: #f8fafc;
      font-size: 13px;
      line-height: 1.45;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.24);
      pointer-events: none;
      white-space: nowrap;
      z-index: 1;
    }}
    .chart-tooltip strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }}
    .tick-label {{
      font-size: 11px;
      fill: #475569;
    }}
    .axis-label {{
      font-size: 12px;
      fill: #0f172a;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{html.escape(page_title)}</h1>
    <p class="summary">
      Source file: {html.escape(str(file_path))}<br />
      Data points parsed: {len(records)}<br />
      Metrics visualized: {html.escape(metric_summary)}
    </p>
    {"".join(metric_sections)}
  </main>
  <script>
    (() => {{
      const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
      const escapeHtml = (value) =>
        String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");

      const parsePoints = (element) => {{
        if (!element._parsedPoints) {{
          element._parsedPoints = JSON.parse(element.dataset.points);
        }}
        return element._parsedPoints;
      }};

      const hideHover = (svg, tooltip) => {{
        const hoverLayer = svg.querySelector(".hover-layer");
        if (hoverLayer) {{
          hoverLayer.setAttribute("visibility", "hidden");
        }}
        tooltip.hidden = true;
      }};

      const formatTooltip = (point, suffix) => {{
        return [
          `Step: ${{escapeHtml(point.step)}}`,
          `Value: ${{escapeHtml(point.display_value)}}${{escapeHtml(suffix)}}`,
        ].join("<br />");
      }};

      document.querySelectorAll(".chart-wrap").forEach((wrap) => {{
        const svg = wrap.querySelector(".metric-svg");
        const tooltip = wrap.querySelector(".chart-tooltip");
        const hoverLayer = svg.querySelector(".hover-layer");
        const hoverGuide = svg.querySelector(".hover-guide");
        const hoverAnchor = svg.querySelector(".hover-anchor");
        const valueSuffix = tooltip.dataset.valueSuffix || "";

        const updateHover = (event) => {{
          const hitArea = event.currentTarget;
          const points = parsePoints(hitArea);
          if (!points.length) {{
            return;
          }}

          const svgPoint = svg.createSVGPoint();
          svgPoint.x = event.clientX;
          svgPoint.y = event.clientY;
          const cursor = svgPoint.matrixTransform(svg.getScreenCTM().inverse());

          let closestPoint = points[0];
          let closestDistance = Math.abs(points[0].x - cursor.x);
          for (let index = 1; index < points.length; index += 1) {{
            const candidate = points[index];
            const distance = Math.abs(candidate.x - cursor.x);
            if (distance < closestDistance) {{
              closestPoint = candidate;
              closestDistance = distance;
            }}
          }}

          const plotTop = Number(svg.dataset.plotTop);
          const plotBottom = Number(svg.dataset.plotBottom);
          const seriesColor = hitArea.dataset.seriesColor || "#2563eb";
          hoverLayer.setAttribute("visibility", "visible");
          hoverGuide.setAttribute("x1", closestPoint.x);
          hoverGuide.setAttribute("x2", closestPoint.x);
          hoverGuide.setAttribute("y1", plotTop);
          hoverGuide.setAttribute("y2", plotBottom);
          hoverAnchor.setAttribute("cx", closestPoint.x);
          hoverAnchor.setAttribute("cy", closestPoint.y);
          hoverAnchor.setAttribute("fill", seriesColor);

          tooltip.innerHTML = formatTooltip(closestPoint, valueSuffix);
          tooltip.hidden = false;

          const viewBox = svg.viewBox.baseVal;
          const xRatio = closestPoint.x / viewBox.width;
          const yRatio = closestPoint.y / viewBox.height;
          const anchorX = xRatio * svg.clientWidth;
          const anchorY = yRatio * svg.clientHeight;

          const tooltipWidth = tooltip.offsetWidth;
          const tooltipHeight = tooltip.offsetHeight;
          let left = anchorX + 14;
          let top = anchorY - tooltipHeight - 14;
          const maxLeft = Math.max(8, wrap.clientWidth - tooltipWidth - 8);
          const maxTop = Math.max(8, wrap.clientHeight - tooltipHeight - 8);

          left = clamp(left, 8, maxLeft);
          if (top < 8) {{
            top = Math.min(maxTop, anchorY + 14);
          }}
          top = clamp(top, 8, maxTop);

          tooltip.style.left = `${{left}}px`;
          tooltip.style.top = `${{top}}px`;
        }};

        svg.querySelectorAll(".metric-hit-area").forEach((hitArea) => {{
          hitArea.addEventListener("pointerenter", updateHover);
          hitArea.addEventListener("pointermove", updateHover);
          hitArea.addEventListener("pointerleave", () => hideHover(svg, tooltip));
        }});

        svg.addEventListener("pointerleave", () => hideHover(svg, tooltip));
      }});
    }})();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    if args.list_metrics:
        list_metrics()
        return

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    runs = parse_training_log(input_file)
    selected_metrics = resolve_metrics(args.metrics, runs)

    output_path = Path(args.output) if args.output else input_file.with_name(f"{input_file.stem}_metrics.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_report = build_html_report(
        file_path=input_file,
        records=runs,
        selected_metrics=selected_metrics,
        title=args.title,
    )
    output_path.write_text(html_report, encoding="utf-8")

    metric_names = ", ".join(metric.key for metric in selected_metrics)
    print(f"Parsed {len(runs)} data points from a single training log.")
    print(f"Metrics: {metric_names}")
    print(f"Saved HTML to: {output_path}")

    if args.show:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
