from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

import torch
import numpy as np

# 确保可以引用到 online_self_healing 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import plotly.graph_objects as go
from online_self_healing.node import NodeElement
from online_self_healing.orbit import SatelliteTracker
from online_self_healing.consensus_engine.engine import BlockchainConsensusStateMachine

class ConstellationExplorer:
    """星座全量数据浏览器：完整展示 11 个性能指标、16 个环境参数及节点模型指纹"""
    
    def __init__(self, output_root_dir: Optional[Union[str, Path]] = None):
        self.engine = BlockchainConsensusStateMachine(output_root_dir=output_root_dir)
        
    def _extract_model_stats(self, pth_path: Path) -> Optional[Dict[str, Any]]:
        """从 .pth 提取详细权重特征"""
        if not pth_path.exists(): return None
        try:
            state_dict = torch.load(pth_path, map_location="cpu")
            if isinstance(state_dict, dict) and any(k in state_dict for k in ["state_dict", "model_state_dict"]):
                state_dict = state_dict.get("state_dict") or state_dict.get("model_state_dict")
            all_w = [v.detach().numpy().flatten() for v in state_dict.values() if isinstance(v, torch.Tensor)]
            if not all_w: return None
            comb = np.concatenate(all_w)
            return {
                "mean": float(np.mean(comb)), "std": float(np.std(comb)),
                "max": float(np.max(comb)), "min": float(np.min(comb)),
                "count": int(len(comb)), "layers": len(state_dict.keys())
            }
        except: return {"error": "Load Failed"}

    def get_full_constellation_report(self, output_selection_input: Any) -> Dict[str, Any]:
        paths = self.engine.output_reader.resolve_paths(output_selection_input)
        report = self.engine.process_output_selection(output_selection_input)
        raw_summary = self.engine.output_reader.load_summary_record(output_selection_input)
        catalog = self.engine._get_catalog(report.fail_env.ConstellationConfig)
        
        diagnosed = {rec.SID: rec for rec in report.satellite_records}
        full_data = []
        
        print(f"正在准备全网数据 (Scenario: {paths.model_dir.name})...")
        for name in catalog.satellite_names:
            node = {"SID": name, "Health": 1.0, "Risk": "低失效风险", "Fail": 0.0, "Attack": "None", "IsFailed": False, "Stats": None}
            if name in diagnosed:
                r = diagnosed[name]
                node.update({"Health": r.HealthScore, "Risk": r.RiskLevel, "Fail": r.FailScore, "Attack": r.AttackTypeLabel, "IsFailed": True})
            pth = paths.model_dir / f"{name}.pth"
            if pth.exists(): node["Stats"] = self._extract_model_stats(pth)
            full_data.append(node)
            
        return {"id": report.scenario_id, "summary": raw_summary, "nodes": full_data, "tle": catalog.tle_path}

    def visualize(self, selection: Any, save_path: Optional[str] = None):
        """生成交互式可视化报告，展示完整的 11+16 指标"""
        report = self.get_full_constellation_report(selection)
        summary = report["summary"]
        
        # 定义完整的指标列表
        ENV_PARAMS_LIST = [
            "ConstellationConfig", "DegradedEdgeRatio", "EdgeDisconnectRatio",
            "EdgeBandwidthMeanDecreaseRatio", "EdgeBandwidthDecreaseStd", "PoissonRate",
            "MeanIntervalTime", "PacketGenerationInterval", "PacketSizeMean", "PacketSizeStd",
            "StateObservationAttack_level", "ActionAttack_level", "RewardAttack_level",
            "StateTransferAttack_level", "ExperiencePoolAttack_level", "ModelTampAttack_level"
        ]
        
        PERF_METRICS_LIST = [
            "PacketLossRate", "NetworkThroughput", "BandwidthUtilization",
            "AvgPacketNodeVisits", "CumulativeReward", "AverageInferenceTime",
            "AverageE2eDelay", "AverageHopCount", "AverageComputingRatio",
            "ComputingWaitingTime", "AverageEndingReward"
        ]
        
        fig = go.Figure()

        # 绘制地球
        r_earth = 6371
        phi, theta = np.mgrid[0.0:2.0*np.pi:50j, 0.0:np.pi:50j]
        xe = r_earth * np.sin(theta) * np.cos(phi)
        ye = r_earth * np.sin(theta) * np.sin(phi)
        ze = r_earth * np.cos(theta)
        fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale='Blues', opacity=0.15, showscale=False, hoverinfo='skip', name="Earth"))

        # 加载坐标
        tracker = SatelliteTracker(report["tle"])
        pos_dict = tracker.generate_satellite_dict("2024-01-01 12:00:00")
        
        groups = {
            "高失效风险": {"c": "#ef4444", "nodes": []},
            "中/批量风险": {"c": "#f59e0b", "nodes": []},
            "低失效风险": {"c": "#10b981", "nodes": []}
        }

        for n in report["nodes"]:
            p = pos_dict.get(n["SID"])
            if not p: continue
            x, y, z = p.eci_position_km
            
            key = "高失效风险" if n["Risk"] == "高失效风险" else ("中/批量风险" if "中" in n["Risk"] else "低失效风险")
            
            detail = {
                "node": n, 
                "pos": {"X": f"{x:.2f} km", "Y": f"{y:.2f} km", "Z": f"{z:.2f} km"},
                "mstats": n["Stats"],
                "env": {k: summary.get(k, "N/A") for k in ENV_PARAMS_LIST},
                "perf": {k: summary.get(k, "N/A") for k in PERF_METRICS_LIST}
            }
            groups[key]["nodes"].append((x, y, z, n["SID"], json.dumps(detail, ensure_ascii=False)))

        for label, g in groups.items():
            if not g["nodes"]: continue
            xs, ys, zs, sids, cdata = zip(*g["nodes"])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='markers',
                marker=dict(size=4, color=g["c"], opacity=0.8),
                name=label, text=sids, customdata=cdata,
                hovertemplate="<b>%{text}</b><br>点击查看完整数据<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                bgcolor="rgba(0,0,0,0)",
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5, bgcolor="rgba(15,23,42,0.5)")
        )

        if save_path:
            save_file = Path(save_path).resolve()
            save_file.parent.mkdir(parents=True, exist_ok=True)
            plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='viz', config={'responsive': True, 'displayModeBar': False})
            
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>iSatCR-V1 全景监控终端</title>
                <style>
                    * {{ box-sizing: border-box; }}
                    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #020617; color: #f1f5f9; font-family: 'Inter', sans-serif; overflow: hidden; }}
                    .layout {{ display: flex; width: 100vw; height: 100vh; }}
                    #viz-area {{ flex: 7.5; height: 100vh; position: relative; background: radial-gradient(circle at center, #111827 0%, #020617 100%); }}
                    #viz {{ width: 100%; height: 100%; }}
                    .plotly-graph-div {{ height: 100vh !important; width: 100% !important; }}

                    #panel {{ flex: 2.5; min-width: 450px; height: 100vh; background: #0f172a; border-left: 1px solid #1e293b; padding: 25px; overflow-y: auto; box-shadow: -10px 0 50px rgba(0,0,0,0.8); z-index: 100; }}
                    h2 {{ font-size: 1.4rem; margin: 0 0 10px 0; border-left: 4px solid #38bdf8; padding-left: 12px; font-weight: 800; }}
                    .sub {{ color: #64748b; font-size: 0.8rem; margin-bottom: 20px; }}
                    h3 {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #38bdf8; margin: 25px 0 12px 0; display: flex; align-items: center; gap: 8px; }}
                    h3::after {{ content: ''; flex: 1; height: 1px; background: #334155; }}
                    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 15px; margin-bottom: 12px; }}
                    .row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #0f172a; font-size: 0.85rem; }}
                    .row:last-child {{ border-bottom: none; }}
                    .lab {{ color: #94a3b8; max-width: 60%; word-break: break-all; }}
                    .val {{ font-weight: 600; font-family: 'JetBrains Mono', monospace; text-align: right; }}
                    .tag {{ font-size: 0.7rem; font-weight: 800; padding: 2px 8px; border-radius: 5px; }}
                    .tag-err {{ background: #7f1d1d; color: #fecaca; }}
                    .tag-ok {{ background: #064e3b; color: #d1fae5; }}
                    #welcome {{ text-align: center; margin-top: 15vh; color: #475569; }}
                    ::-webkit-scrollbar {{ width: 5px; }}
                    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 10px; }}
                </style>
            </head>
            <body>
                <div class="layout">
                    <div id="viz-area">{plotly_html}</div>
                    <div id="panel">
                        <h2>自愈诊断监控台</h2>
                        <div class="sub">Report ID: {report['id']}</div>
                        <div id="welcome"><div style="font-size: 5rem; margin-bottom: 20px;">📡</div><p>请在 3D 视图中点击卫星节点<br>展示全量性能、环境与模型参数</p></div>
                        <div id="details" style="display:none;">
                            <h3>节点基本信息 (Identity & Position)</h3>
                            <div class="card" id="box-id"></div>
                            <h3>本地模型权重分析 (Model Weights)</h3>
                            <div class="card" id="box-model"></div>
                            <h3>11项输出性能指标 (Performance Metrics)</h3>
                            <div class="card" id="box-perf"></div>
                            <h3>16项环境输入参数 (Environment Params)</h3>
                            <div class="card" id="box-env"></div>
                        </div>
                    </div>
                </div>
                <script>
                    window.onload = function() {{
                        var plot = document.getElementById('viz');
                        plot.on('plotly_click', function(data){{
                            if(!data.points.length) return;
                            var raw = data.points[0].customdata;
                            if(Array.isArray(raw)) raw = raw[0];
                            var d = JSON.parse(raw);
                            document.getElementById('welcome').style.display = 'none';
                            document.getElementById('details').style.display = 'block';
                            var f = (v) => (typeof v === 'number') ? (Number.isInteger(v) ? v : v.toFixed(5)) : v;
                            
                            // Basic Info
                            var n = d.node; var p = d.pos;
                            document.getElementById('box-id').innerHTML = `
                                <div class="row"><span class="lab">SID (卫星 ID)</span><span class="val">${{n.SID}}</span></div>
                                <div class="row"><span class="lab">ECI 坐标 (X, Y, Z)</span><span class="val">${{p.X}}, ${{p.Y}}, ${{p.Z}}</span></div>
                                <div class="row"><span class="lab">失效判定得分</span><span class="val" style="color:#fb7185">${{f(n.Fail)}}</span></div>
                                <div class="row"><span class="lab">风险等级</span><span class="tag ${{n.IsFailed?'tag-err':'tag-ok'}}">${{n.Risk}}</span></div>
                                <div class="row"><span class="lab">疑似受攻击类型</span><span class="val">${{n.Attack}}</span></div>
                            `;
                            
                            // Model
                            var m = d.mstats;
                            if(m && !m.error) {{
                                document.getElementById('box-model').innerHTML = `
                                    <div class="row"><span class="lab">权重均值 (Mean)</span><span class="val">${{f(m.mean)}}</span></div>
                                    <div class="row"><span class="lab">权重标准差 (Std)</span><span class="val">${{f(m.std)}}</span></div>
                                    <div class="row"><span class="lab">数值范围 (Min/Max)</span><span class="val">${{f(m.min)}} / ${{f(m.max)}}</span></div>
                                    <div class="row"><span class="lab">总参数量 / 层数</span><span class="val">${{m.count.toLocaleString()}} / ${{m.layers}}</span></div>
                                `;
                            }} else {{
                                document.getElementById('box-model').innerHTML = '<div style="color:#64748b; font-size:0.8rem; text-align:center;">该节点暂无可用本地权重 (.pth)</div>';
                            }}
                            
                            // Performance (11 Metrics)
                            var pH = ''; for(var k in d.perf) pH += `<div class="row"><span class="lab">${{k}}</span><span class="val">${{f(d.perf[k])}}</span></div>`;
                            document.getElementById('box-perf').innerHTML = pH;
                            
                            // Environment (16 Params)
                            var eH = ''; for(var k in d.env) eH += `<div class="row"><span class="lab">${{k}}</span><span class="val">${{f(d.env[k])}}</span></div>`;
                            document.getElementById('box-env').innerHTML = eH;
                            document.getElementById('panel').scrollTop = 0;
                        }});
                    }};
                </script>
            </body>
            </html>
            """
            with open(save_file, "w", encoding="utf-8") as f: f.write(full_html)
            print(f"可视化报告已完整更新 (11性能+16环境): {save_file}")
        else: fig.show()

if __name__ == "__main__":
    explorer = ConstellationExplorer()
    selection = {"output_constellation_index": 0, "round_index": 25, "test_id": 401}
    output_html = PROJECT_ROOT / "result" / "explorer" / "detailed_constellation_view.html"
    explorer.visualize(selection, save_path=str(output_html))
