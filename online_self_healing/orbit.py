from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

from sgp4.api import SGP4_ERRORS, Satrec, jday


TimeInput = Union[datetime, str]


@dataclass(frozen=True)
class SatelliteState:
    name: str
    orbit_altitude: int
    orbit_number: int
    sat_number: int
    eci_position_km: Tuple[float, float, float]
    eci_velocity_km_s: Tuple[float, float, float]

    @property
    def sequence_num(self) -> Tuple[int, int, int]:
        return (self.orbit_altitude, self.orbit_number, self.sat_number)


def parse_satellite_name(name: str) -> Tuple[int, int, int]:
    parts = str(name).split("_")
    if len(parts) != 4 or parts[0] != "Satellite":
        raise ValueError(
            f"Satellite name '{name}' does not match the required pattern "
            "Satellite_<orbit_altitude>_<orbit_number>_<sat_number>"
        )
    return int(parts[1]), int(parts[2]), int(parts[3])


def ensure_datetime_utc(current_time: TimeInput) -> datetime:
    if isinstance(current_time, datetime):
        dt = current_time
    elif isinstance(current_time, str):
        text = current_time.strip()
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        )
        dt = None
        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            dt = datetime.fromisoformat(text)
    else:
        raise TypeError("CurrentTime must be a datetime or time string")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SatelliteTracker:
    def __init__(self, tle_filepath: Union[str, Path]):
        self.tle_filepath = Path(tle_filepath)
        self.satellites = self._load_satellites()

    def _load_satellites(self) -> Dict[str, Satrec]:
        if not self.tle_filepath.exists():
            raise FileNotFoundError(f"TLE file not found: {self.tle_filepath}")

        with self.tle_filepath.open("r", encoding="utf-8") as file:
            tle_lines = [line.strip() for line in file.readlines() if line.strip()]

        if len(tle_lines) % 3 != 0:
            raise ValueError(
                f"TLE file {self.tle_filepath} must contain 3-line groups of "
                "name + line1 + line2"
            )

        satellites: Dict[str, Satrec] = {}
        for index in range(0, len(tle_lines), 3):
            sat_name = tle_lines[index]
            line1 = tle_lines[index + 1]
            line2 = tle_lines[index + 2]
            satellites[sat_name] = Satrec.twoline2rv(line1, line2)
        return satellites

    def _propagate(self, sat_name: str, satrec: Satrec, current_time: TimeInput) -> SatelliteState:
        dt_utc = ensure_datetime_utc(current_time)
        jd, fr = jday(
            dt_utc.year,
            dt_utc.month,
            dt_utc.day,
            dt_utc.hour,
            dt_utc.minute,
            dt_utc.second + dt_utc.microsecond / 1_000_000.0,
        )
        error_code, position, velocity = satrec.sgp4(jd, fr)
        if error_code != 0:
            reason = SGP4_ERRORS.get(error_code, "Unknown SGP4 error")
            raise RuntimeError(f"SGP4 propagation failed for {sat_name}: [{error_code}] {reason}")

        orbit_altitude, orbit_number, sat_number = parse_satellite_name(sat_name)
        return SatelliteState(
            name=sat_name,
            orbit_altitude=orbit_altitude,
            orbit_number=orbit_number,
            sat_number=sat_number,
            eci_position_km=tuple(float(value) for value in position),
            eci_velocity_km_s=tuple(float(value) for value in velocity),
        )

    def generate_satellite_dict(self, current_time: TimeInput) -> Dict[str, SatelliteState]:
        return {
            sat_name: self._propagate(sat_name, satrec, current_time)
            for sat_name, satrec in self.satellites.items()
        }

    def get_max_orbit_number(self) -> int:
        return max(parse_satellite_name(sat_name)[1] for sat_name in self.satellites)

    def get_max_satellite_number(self) -> int:
        return max(parse_satellite_name(sat_name)[2] for sat_name in self.satellites)

    def satellite_names(self) -> Iterable[str]:
        return self.satellites.keys()
