"""Single project-wide Memory Guard thresholds and classifier."""

from dataclasses import dataclass
import json
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class GuardThresholds:
    warning_available_gib: float = 2.0
    critical_available_gib: float = 0.5
    warning_commit_percent: float = 85.0
    critical_commit_percent: float = 95.0
    warning_load_percent: float = 85.0
    critical_load_percent: float = 95.0

    def __post_init__(self) -> None:
        if self.critical_available_gib < 0 or self.warning_available_gib <= self.critical_available_gib:
            raise ValueError("Available-memory thresholds must satisfy 0 <= critical < warning.")
        for value in (
            self.warning_commit_percent,
            self.critical_commit_percent,
            self.warning_load_percent,
            self.critical_load_percent,
        ):
            if not 0 <= value <= 100:
                raise ValueError("Percent thresholds must be between 0 and 100.")
        if self.critical_commit_percent < self.warning_commit_percent:
            raise ValueError("Critical commit threshold must be >= warning threshold.")
        if self.critical_load_percent < self.warning_load_percent:
            raise ValueError("Critical load threshold must be >= warning threshold.")

    def classify(self, *, available_bytes: int, physical_memory_load_percent: int,
                 commit_used_percent: float) -> str:
        if (available_bytes < self.critical_available_gib * GIB
                or commit_used_percent >= self.critical_commit_percent
                or physical_memory_load_percent >= self.critical_load_percent):
            return "critical"
        if (available_bytes < self.warning_available_gib * GIB
                or commit_used_percent >= self.warning_commit_percent
                or physical_memory_load_percent >= self.warning_load_percent):
            return "warning"
        return "normal"


def load_guard_thresholds(config_path: str | Path | None = None) -> GuardThresholds:
    """Read the one authoritative threshold set from config/settings.json."""
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        values = settings.get("stage_2", {})
    except (OSError, json.JSONDecodeError):
        return GuardThresholds()
    return GuardThresholds(
        warning_available_gib=float(values.get("warning_available_memory_mib", 2048)) / 1024,
        critical_available_gib=float(values.get("critical_available_memory_mib", 512)) / 1024,
        warning_commit_percent=float(values.get("warning_commit_percent", 85)),
        critical_commit_percent=float(values.get("critical_commit_percent", 95)),
        warning_load_percent=float(values.get("warning_physical_memory_load_percent", 85)),
        critical_load_percent=float(values.get("critical_physical_memory_load_percent", 95)),
    )


DEFAULT_GUARD_THRESHOLDS = load_guard_thresholds()
