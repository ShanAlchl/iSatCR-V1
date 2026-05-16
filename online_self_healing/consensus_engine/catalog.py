from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from constellation_tle_order import resolve_constellation_tle_path
from ..orbit import parse_satellite_name


def _load_tle_satellite_names(tle_path: Path) -> Tuple[str, ...]:
    if not tle_path.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_path}")

    with tle_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"TLE file {tle_path} must contain repeated 3-line groups of "
            "satellite name + line1 + line2."
        )

    return tuple(lines[index] for index in range(0, len(lines), 3))


@dataclass(frozen=True)
class ConstellationCatalog:
    constellation_config: int
    tle_path: Path
    satellite_names: Tuple[str, ...]
    adjacency: Dict[str, frozenset[str]]
    name_to_one_based_index: Dict[str, int]

    @classmethod
    def from_constellation_config(
        cls,
        constellation_config: int,
        *,
        satellite_data_dir: Optional[Union[str, Path]] = None,
    ) -> "ConstellationCatalog":
        data_dir = (
            Path(satellite_data_dir)
            if satellite_data_dir is not None
            else Path(__file__).resolve().parent.parent / "Satellite_Data"
        )
        config_index = int(constellation_config)
        tle_path = resolve_constellation_tle_path(data_dir, config_index)
        satellite_names = _load_tle_satellite_names(tle_path)
        adjacency = cls._build_fixed_adjacency(satellite_names)
        name_to_one_based_index = {
            name: index for index, name in enumerate(satellite_names, start=1)
        }
        return cls(
            constellation_config=config_index,
            tle_path=tle_path,
            satellite_names=satellite_names,
            adjacency=adjacency,
            name_to_one_based_index=name_to_one_based_index,
        )

    @staticmethod
    def _build_fixed_adjacency(
        satellite_names: Iterable[str],
    ) -> Dict[str, frozenset[str]]:
        names = tuple(satellite_names)
        parsed = {name: parse_satellite_name(name) for name in names}
        lookup = {sequence_num: name for name, sequence_num in parsed.items()}
        max_orbit_number = max(sequence_num[1] for sequence_num in parsed.values())
        max_sat_number = max(sequence_num[2] for sequence_num in parsed.values())

        adjacency: Dict[str, Set[str]] = {name: set() for name in names}

        for name, (orbit_altitude, orbit_number, sat_number) in parsed.items():
            prev_sat_number = sat_number - 1 if sat_number != 1 else max_sat_number
            next_sat_number = (sat_number % max_sat_number) + 1
            for neighbor_key in (
                (orbit_altitude, orbit_number, prev_sat_number),
                (orbit_altitude, orbit_number, next_sat_number),
            ):
                neighbor = lookup.get(neighbor_key)
                if neighbor is None or neighbor == name:
                    continue
                adjacency[name].add(neighbor)
                adjacency[neighbor].add(name)

        for name, (orbit_altitude, orbit_number, sat_number) in parsed.items():
            next_orbit_number = (orbit_number % max_orbit_number) + 1
            next_sat_number = (
                sat_number
                if next_orbit_number % 2 == 0
                else (sat_number - 1 if sat_number != 1 else max_sat_number)
            )
            neighbor = lookup.get((orbit_altitude, next_orbit_number, next_sat_number))
            if neighbor is None or neighbor == name:
                continue
            adjacency[name].add(neighbor)
            adjacency[neighbor].add(name)

        return {name: frozenset(neighbors) for name, neighbors in adjacency.items()}

    def resolve_satellite_id(
        self,
        raw_sid: Union[int, str],
        *,
        sid_index_base: int = 1,
    ) -> str:
        if isinstance(raw_sid, str) and raw_sid.startswith("Satellite_"):
            if raw_sid not in self.adjacency:
                raise KeyError(
                    f"Satellite identifier '{raw_sid}' was not found in {self.tle_path.name}."
                )
            return raw_sid

        raw_index = int(raw_sid)
        one_based_index = raw_index if sid_index_base == 1 else raw_index + 1
        if one_based_index < 1 or one_based_index > len(self.satellite_names):
            raise IndexError(
                f"Satellite index {raw_sid} is out of range for {self.tle_path.name}. "
                f"Expected {sid_index_base}-based indices within "
                f"{0 if sid_index_base == 0 else 1}.."
                f"{len(self.satellite_names) - (1 if sid_index_base == 0 else 0)}."
            )
        return self.satellite_names[one_based_index - 1]

    def neighbors_of(self, satellite_id: str) -> frozenset[str]:
        return self.adjacency.get(satellite_id, frozenset())
