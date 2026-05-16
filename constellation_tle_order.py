from __future__ import annotations

from pathlib import Path
from typing import Union


CONSTELLATION_TLE_FILENAMES = (
    "Delta_24_18_50_1150_5.txt",
    "Delta_72_22_53_550_11.txt",
    "Delta_60_60_58_500_23.txt",
    "Star_16_10_98_600_7.txt",
    "Star_6_10_86_1100_5.txt",
)


def ordered_constellation_tle_paths(satellite_data_dir: Union[str, Path]) -> tuple[Path, ...]:
    data_dir = Path(satellite_data_dir)
    paths = tuple(data_dir / filename for filename in CONSTELLATION_TLE_FILENAMES)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing configured TLE files under {data_dir}: {missing}"
        )
    return paths


def resolve_constellation_tle_path(
    satellite_data_dir: Union[str, Path],
    constellation_config: int,
) -> Path:
    paths = ordered_constellation_tle_paths(satellite_data_dir)
    config_index = int(constellation_config)
    if config_index < 0 or config_index >= len(paths):
        raise IndexError(
            "ConstellationConfig is out of range for the configured TLE catalog. "
            f"Received {config_index}, available range is 0..{len(paths) - 1}."
        )
    return paths[config_index]
