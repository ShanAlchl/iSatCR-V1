#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = r"train\train_NewDDQN_dueling_shuffle.yaml"
MODEL_ROOT = Path(r".\low_medium_models")
ATTACK_FIELDS = [
    ("StateObservationAttack_level", "StateObservationAttack"),
    ("ActionAttack_level", "ActionAttack"),
    ("RewardAttack_level", "RewardAttack"),
    ("StateTransferAttack_level", "StateTransferAttack"),
    ("ExperiencePoolAttack_level", "ExperiencePoolAttack"),
    ("ModelTampAttack_level", "ModelTampAttack"),
]

INT_PATTERN = re.compile(r"^[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
SAFE_PATH_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成实验组合并依次运行 PRC.py。")
    parser.add_argument("--env-md", required=True, help="实验环境 md 文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="yaml 配置文件路径")
    parser.add_argument(
        "--force-fresh-start",
        action="store_true",
        help="为每次实验自动生成新的 checkpoint 路径，强制从零开始训练而不是续训旧模型",
    )
    return parser.parse_args()


def resolve_path(path_str, project_root):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_scalar(token):
    token = token.strip()
    if not token:
        return token
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()

    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if INT_PATTERN.fullmatch(token):
        return int(token)
    if FLOAT_PATTERN.fullmatch(token):
        return float(token)
    return token


def parse_experiment_md(md_path):
    params = {}
    with open(md_path, "r", encoding="utf-8-sig") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("```"):
                continue

            line = re.sub(r"^>\s*", "", line)
            line = re.sub(r"^[-*+]\s+", "", line)
            line = re.sub(r"^\d+\.\s+", "", line)
            line = line.strip()
            if line.startswith("`") and line.endswith("`") and len(line) >= 2:
                line = line[1:-1].strip()

            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", line)
            if not match:
                continue

            key, raw_values = match.group(1), match.group(2).strip()
            try:
                value_cells = next(csv.reader([raw_values], skipinitialspace=True))
            except Exception as exc:
                raise ValueError(f"解析 md 失败，行 {lineno}: {raw_line.rstrip()}") from exc

            values = []
            for cell in value_cells:
                cell = cell.strip()
                if not cell:
                    continue
                if cell.startswith("`") and cell.endswith("`") and len(cell) >= 2:
                    cell = cell[1:-1].strip()
                values.append(parse_scalar(cell))

            if not values:
                continue
            if key in params:
                print(f"[Warn] 参数 {key} 在 md 中重复定义，使用后一次定义。", flush=True)
            params[key] = values
    return params


def format_name_part(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")

    text = str(value).strip()
    if INT_PATTERN.fullmatch(text):
        return str(int(text))
    if FLOAT_PATTERN.fullmatch(text):
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
        return format(as_float, "g")
    return text


def sanitize_path_token(value):
    sanitized = SAFE_PATH_TOKEN_PATTERN.sub("_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"


def parse_attack_level(value, field_name):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} 需要是整数，当前值: {value}")
        return int(value)

    text = str(value).strip()
    if not text:
        return 0
    as_float = float(text)
    if not as_float.is_integer():
        raise ValueError(f"{field_name} 需要是整数，当前值: {value}")
    return int(as_float)


def count_enabled_attacks(env_cfg):
    enabled_count = 0
    for field_name, _ in ATTACK_FIELDS:
        level = parse_attack_level(env_cfg.get(field_name, 0), field_name)
        if level > 0:
            enabled_count += 1
    return enabled_count


def is_attack_combination_valid(combo, template_env):
    # 约束：最终生效配置中，最多允许 1 个攻击字段 > 0
    merged_env = dict(template_env)
    merged_env.update(combo)
    return count_enabled_attacks(merged_env) <= 1


def build_attack_descriptor(env_cfg):
    parts = []
    for field_name, attack_name in ATTACK_FIELDS:
        level = parse_attack_level(env_cfg.get(field_name, 0), field_name)
        if level > 0:
            parts.append(f"attack_{attack_name}_level_{format_name_part(level)}")
    if not parts:
        return "attack_none"
    return "__".join(parts)


def build_artifact_basename(config, attack_descriptor_override=None):
    env_cfg = config.setdefault("environment", {})
    agent_cfg = config.setdefault("agent", {})
    general_cfg = config.setdefault("general", {})

    traffic_profile = str(env_cfg.get("TrafficProfile", "")).strip().strip('"').strip("'")
    if not traffic_profile:
        raise ValueError("TrafficProfile 不能为空，无法生成路径和文件名。")

    if "ConstellationConfig" not in env_cfg:
        raise ValueError("ConstellationConfig 缺失，无法生成路径和文件名。")

    if "duration" not in general_cfg:
        raise ValueError("general.duration 缺失，无法生成路径和文件名。")

    if "batch_size" not in agent_cfg:
        raise ValueError("agent.batch_size 缺失，无法生成路径和文件名。")

    if "epsilon" not in general_cfg:
        raise ValueError("general.epsilon 缺失，无法生成路径和文件名。")

    parts = [
        f"constellation_{sanitize_path_token(format_name_part(env_cfg.get('ConstellationConfig')))}",
        f"trafficprofile_{sanitize_path_token(traffic_profile)}",
        attack_descriptor_override or build_attack_descriptor(env_cfg),
        f"duration_{sanitize_path_token(format_name_part(general_cfg.get('duration')))}",
        f"batchsize_{sanitize_path_token(format_name_part(agent_cfg.get('batch_size')))}",
        f"epsilon_{sanitize_path_token(format_name_part(general_cfg.get('epsilon')))}",
    ]
    return "__".join(parts)


def update_auto_fields(config, force_fresh_start=False):
    env_cfg = config.setdefault("environment", {})
    agent_cfg = config.setdefault("agent", {})

    traffic_profile = str(env_cfg.get("TrafficProfile", "")).strip().strip('"').strip("'")
    if not traffic_profile:
        raise ValueError("TrafficProfile 不能为空，无法生成路径和文件名。")
    env_cfg["TrafficProfile"] = traffic_profile

    if "ConstellationConfig" not in env_cfg:
        raise ValueError("ConstellationConfig 缺失，无法生成路径和文件名。")
    sharing_mode = str(agent_cfg.get("agent_sharing_mode", "shared")).strip().lower()
    artifact_basename = build_artifact_basename(config)
    checkpoint_basename = artifact_basename
    save_training_data = f"{artifact_basename}.txt"
    
    model_path = MODEL_ROOT / f"{checkpoint_basename}.pth"
    independent_model_dir = MODEL_ROOT / f"{checkpoint_basename}__independent"

    if force_fresh_start:
        agent_cfg["bootstrap_model_path"] = str(model_path)
    elif sharing_mode == "independent":
        bootstrap_basename = build_artifact_basename(config, attack_descriptor_override="attack_none")
        agent_cfg["bootstrap_model_path"] = str(MODEL_ROOT / f"{bootstrap_basename}.pth")

    agent_cfg["independent_model_dir"] = str(independent_model_dir)
    agent_cfg["model_path"] = str(model_path)
    env_cfg["SaveTrainingData"] = save_training_data
    return {
        "experiment_name": Path(save_training_data).stem,
        "model_path": str(model_path),
        "independent_model_dir": str(independent_model_dir),
        "bootstrap_model_path": str(agent_cfg.get("bootstrap_model_path", "")),
        "sharing_mode": sharing_mode,
    }


def collect_fresh_start_conflicts(project_root, artifact_info):
    conflict_items = []

    model_path = resolve_path(artifact_info["model_path"], project_root)
    if model_path.exists():
        conflict_items.append(f"model_path already exists: {model_path}")

    if artifact_info["sharing_mode"] == "independent":
        independent_model_dir = resolve_path(artifact_info["independent_model_dir"], project_root)
        if independent_model_dir.exists():
            conflict_items.append(f"independent_model_dir already exists: {independent_model_dir}")

    save_training_data_path = project_root / "training_process_data" / f"{artifact_info['experiment_name']}.txt"
    if save_training_data_path.exists():
        conflict_items.append(f"SaveTrainingData already exists: {save_training_data_path}")

    return conflict_items


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def config_arg_for_subprocess(config_path, project_root):
    try:
        return str(config_path.relative_to(project_root))
    except ValueError:
        return str(config_path)


def run_batch(project_root, env_md_path, yaml_path, force_fresh_start=False):
    if not env_md_path.exists():
        raise FileNotFoundError(f"实验环境文件不存在: {env_md_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml 文件不存在: {yaml_path}")

    param_space = parse_experiment_md(env_md_path)
    keys = list(param_space.keys())
    value_lists = [param_space[k] for k in keys]
    candidate_total = math.prod(len(v) for v in value_lists) if value_lists else 1

    print(f"实验环境文件: {env_md_path}", flush=True)
    print(f"配置模板: {yaml_path}", flush=True)
    print(f"参数数量: {len(keys)}", flush=True)
    print(f"候选组合总数: {candidate_total}", flush=True)
    if keys:
        print(f"参数顺序: {keys}", flush=True)
    else:
        print("未解析到参数，按模板默认配置执行 1 次。", flush=True)
    if force_fresh_start:
        print("已启用强制从零开始：每次实验都会使用新的 checkpoint 路径。", flush=True)

    original_yaml_bytes = yaml_path.read_bytes()
    template_config = load_yaml(yaml_path)
    template_env = template_config.setdefault("environment", {})
    config_arg = config_arg_for_subprocess(yaml_path, project_root)

    # 先统计有效组合数，保证进度条中的总数准确
    valid_total = 0
    skipped_total = 0
    count_iter = itertools.product(*value_lists) if value_lists else [tuple()]
    for combo_values in count_iter:
        combo = {k: v for k, v in zip(keys, combo_values)}
        if is_attack_combination_valid(combo, template_env):
            valid_total += 1
        else:
            skipped_total += 1

    print(f"有效组合数（将执行）: {valid_total}", flush=True)
    print(f"过滤掉的无效组合数: {skipped_total}", flush=True)

    if valid_total == 0:
        print("没有满足攻击约束（同时最多一个攻击字段 > 0）的组合，脚本结束。", flush=True)
        return 0

    success_count = 0
    failure_items = []
    executed_count = 0
    interrupted = False
    try:
        combo_iter = itertools.product(*value_lists) if value_lists else [tuple()]
        run_index = 0
        for combo_values in combo_iter:
            combo = {k: v for k, v in zip(keys, combo_values)}
            if not is_attack_combination_valid(combo, template_env):
                continue

            run_index += 1
            executed_count = run_index
            current_config = copy.deepcopy(template_config)
            current_env = current_config.setdefault("environment", {})
            for k, v in combo.items():
                current_env[k] = v

            try:
                artifact_info = update_auto_fields(current_config, force_fresh_start=force_fresh_start)
                experiment_name = artifact_info["experiment_name"]
                if force_fresh_start:
                    conflict_items = collect_fresh_start_conflicts(project_root, artifact_info)
                    if conflict_items:
                        joined_conflicts = "; ".join(conflict_items)
                        raise FileExistsError(
                            f"fresh-start target already exists for this parameter set; refusing to overwrite. {joined_conflicts}"
                        )
                print("\n" + "=" * 90, flush=True)
                print(f"[{run_index}/{valid_total}] 开始实验: {experiment_name}", flush=True)
                print(f"参数组合: {combo if combo else '使用模板默认配置'}", flush=True)
                print(f"model_path: {artifact_info['model_path']}", flush=True)
                if artifact_info["sharing_mode"] == "independent":
                    print(f"bootstrap_model_path: {artifact_info['bootstrap_model_path']}", flush=True)
                    print(f"independent_model_dir: {artifact_info['independent_model_dir']}", flush=True)
                elif force_fresh_start:
                    print("fresh-start mode: target names are parameter-based only; existing targets will cause an error", flush=True)

                dump_yaml(yaml_path, current_config)
                completed = subprocess.run(
                    [sys.executable, "PRC.py", "--config", config_arg],
                    cwd=str(project_root),
                    check=False,
                )
                if completed.returncode == 0:
                    success_count += 1
                    print(f"[{run_index}/{valid_total}] 实验成功", flush=True)
                else:
                    error_msg = f"实验进程退出码: {completed.returncode}"
                    failure_items.append(
                        {
                            "index": run_index,
                            "experiment_name": experiment_name,
                            "combo": combo,
                            "error": error_msg,
                        }
                    )
                    print(f"[{run_index}/{valid_total}] 实验失败: {error_msg}", flush=True)
            except Exception as exp:
                error_msg = f"{type(exp).__name__}: {exp}"
                fallback_name = str(current_env.get("SaveTrainingData", "unknown"))
                failure_items.append(
                    {
                        "index": run_index,
                        "experiment_name": fallback_name,
                        "combo": combo,
                        "error": error_msg,
                    }
                )
                print(f"[{run_index}/{valid_total}] 实验失败: {error_msg}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n检测到手动中断，停止提交后续实验。", flush=True)
    finally:
        yaml_path.write_bytes(original_yaml_bytes)
        print("\n已恢复原始 yaml 模板内容。", flush=True)

    failure_count = len(failure_items)
    print("\n" + "=" * 90, flush=True)
    print("批量实验汇总", flush=True)
    print(f"候选组合总数: {candidate_total}", flush=True)
    print(f"有效组合数（将执行）: {valid_total}", flush=True)
    print(f"过滤掉的无效组合数: {skipped_total}", flush=True)
    print(f"已执行: {executed_count}", flush=True)
    print(f"成功: {success_count}", flush=True)
    print(f"失败: {failure_count}", flush=True)
    if failure_items:
        print("失败组合列表:", flush=True)
        for item in failure_items:
            print(
                f"- [{item['index']}/{valid_total}] {item['experiment_name']} | 参数={item['combo']} | 错误={item['error']}",
                flush=True,
            )
    else:
        print("失败组合列表: 无", flush=True)

    if interrupted:
        return 130
    return 0 if failure_count == 0 else 1


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    env_md_path = resolve_path(args.env_md, project_root)
    yaml_path = resolve_path(args.config, project_root)

    try:
        return run_batch(project_root, env_md_path, yaml_path, force_fresh_start=args.force_fresh_start)
    except Exception as exc:
        print(f"批量实验脚本异常退出: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
