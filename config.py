"""Typed configuration loading and persistence for Virtual Mouse.

The application keeps runtime settings in ``config.json``.  This module owns
validation and conversion so the camera, tracking, gesture, and input layers
can work with small, typed configuration objects instead of passing loosely
typed dictionaries around.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


@dataclass
class ScreenRegion:
    """Normalized portion of the screen used for cursor mapping.

    All values are fractions of the complete screen.  For example, ``x=0.0``
    and ``width=0.5`` maps the left half of the screen.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def validate(self) -> None:
        """Validate the region's bounds and dimensions."""
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ConfigError(f"screen_region.{name} must be a number")
        if self.x < 0.0 or self.y < 0.0:
            raise ConfigError("screen_region.x and screen_region.y cannot be negative")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ConfigError("screen_region width and height must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ConfigError("screen_region must fit within the normalized screen")


@dataclass
class CameraSettings:
    """Webcam capture settings."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    flip: bool = True
    reconnect_attempts: int = 3
    reconnect_backoff_seconds: float = 1.0

    def validate(self) -> None:
        """Validate camera settings."""
        if self.index < 0:
            raise ConfigError("camera.index cannot be negative")
        for name in ("width", "height", "fps"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"camera.{name} must be positive")
        if self.reconnect_attempts < 0:
            raise ConfigError("camera.reconnect_attempts cannot be negative")
        if not math.isfinite(float(self.reconnect_backoff_seconds)):
            raise ConfigError("camera.reconnect_backoff_seconds must be finite")
        if self.reconnect_backoff_seconds < 0.0:
            raise ConfigError("camera.reconnect_backoff_seconds cannot be negative")


@dataclass
class MouseSettings:
    """Cursor movement and screen mapping settings."""

    smoothing_factor: float = 5.0
    movement_scale: float = 1.0
    dead_zone_pixels: float = 1.5
    screen_region: ScreenRegion = field(default_factory=ScreenRegion)
    smoothing_backend: str = "exponential"
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.05
    one_euro_dcutoff: float = 1.0

    def validate(self) -> None:
        """Validate cursor movement settings."""
        for name in ("smoothing_factor", "movement_scale", "dead_zone_pixels"):
            if not math.isfinite(float(getattr(self, name))):
                raise ConfigError(f"mouse.{name} must be finite")
        if self.smoothing_factor < 1.0:
            raise ConfigError("mouse.smoothing_factor must be at least 1.0")
        if self.movement_scale <= 0.0:
            raise ConfigError("mouse.movement_scale must be positive")
        if self.dead_zone_pixels < 0.0:
            raise ConfigError("mouse.dead_zone_pixels cannot be negative")
        if self.smoothing_backend not in ("exponential", "one_euro"):
            raise ConfigError("mouse.smoothing_backend must be exponential or one_euro")
        for name in ("one_euro_min_cutoff", "one_euro_beta", "one_euro_dcutoff"):
            if not math.isfinite(float(getattr(self, name))):
                raise ConfigError(f"mouse.{name} must be finite")
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"mouse.{name} must be positive")
        self.screen_region.validate()

    @property
    def smoothening(self) -> float:
        """Backward-compatible spelling for the smoothing factor."""
        return self.smoothing_factor

    @smoothening.setter
    def smoothening(self, value: float) -> None:
        """Set the smoothing factor using the legacy spelling."""
        self.smoothing_factor = value


@dataclass
class GestureSettings:
    """Thresholds and feature switches for gesture classification."""

    pinch_threshold: float = 0.055
    fist_tip_wrist_threshold: float = 0.52
    finger_extension_ratio: float = 1.12
    finger_angle_threshold_degrees: float = 150.0
    peace_min_separation: float = 0.035
    scroll_enabled: bool = True
    scroll_deadzone: float = 0.035
    scroll_amount: int = 3
    scroll_cooldown_frames: int = 4
    scroll_hysteresis: float = 0.01
    drag_enabled: bool = True
    drag_hold_frames: int = 18
    drag_release_frames: int = 3

    def validate(self) -> None:
        """Validate gesture thresholds and switches."""
        for name in (
            "pinch_threshold",
            "fist_tip_wrist_threshold",
            "finger_extension_ratio",
            "finger_angle_threshold_degrees",
            "peace_min_separation",
            "scroll_deadzone",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ConfigError(f"gestures.{name} must be finite")
        for name in (
            "pinch_threshold",
            "fist_tip_wrist_threshold",
            "finger_extension_ratio",
            "peace_min_separation",
        ):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"gestures.{name} must be positive")
        if not 0.0 < self.finger_angle_threshold_degrees < 180.0:
            raise ConfigError(
                "gestures.finger_angle_threshold_degrees must be between 0 and 180"
            )
        if self.scroll_deadzone < 0.0:
            raise ConfigError("gestures.scroll_deadzone cannot be negative")
        if self.scroll_amount <= 0:
            raise ConfigError("gestures.scroll_amount must be positive")
        if self.scroll_cooldown_frames < 0:
            raise ConfigError("gestures.scroll_cooldown_frames cannot be negative")
        if not math.isfinite(float(self.scroll_hysteresis)):
            raise ConfigError("gestures.scroll_hysteresis must be finite")
        if self.scroll_hysteresis < 0.0:
            raise ConfigError("gestures.scroll_hysteresis cannot be negative")
        if self.drag_hold_frames < 1:
            raise ConfigError("gestures.drag_hold_frames must be at least 1")
        if self.drag_release_frames < 1:
            raise ConfigError("gestures.drag_release_frames must be at least 1")


@dataclass
class TapSettings:
    """Thresholds and timing for index-finger tap recognition."""

    enabled: bool = True
    velocity_threshold: float = 0.05
    min_frames_for_tap: int = 3
    max_frames_for_tap: int = 5
    history_size: int = 10
    extended_stable_frames: int = 2
    rearm_frames: int = 2
    extended_y_margin: float = 0.01
    min_tip_drop: float = 0.02
    conflict_lock_frames: int = 5
    calibration_step: float = 0.005

    def validate(self) -> None:
        """Validate tap detector timing and sensitivity settings."""
        for name in (
            "velocity_threshold",
            "extended_y_margin",
            "min_tip_drop",
            "calibration_step",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ConfigError(f"tap.{name} must be finite")
        if self.velocity_threshold <= 0.0:
            raise ConfigError("tap.velocity_threshold must be positive")
        if self.min_frames_for_tap < 1:
            raise ConfigError("tap.min_frames_for_tap must be at least 1")
        if self.max_frames_for_tap < self.min_frames_for_tap:
            raise ConfigError(
                "tap.max_frames_for_tap must be at least min_frames_for_tap"
            )
        if self.history_size < self.max_frames_for_tap:
            raise ConfigError(
                "tap.history_size must be at least max_frames_for_tap"
            )
        if self.extended_stable_frames < 1 or self.rearm_frames < 1:
            raise ConfigError("tap stable/rearm frame counts must be at least 1")
        if self.extended_y_margin < 0.0 or self.min_tip_drop < 0.0:
            raise ConfigError("tap y thresholds cannot be negative")
        if self.conflict_lock_frames < 0:
            raise ConfigError("tap.conflict_lock_frames cannot be negative")
        if self.calibration_step <= 0.0:
            raise ConfigError("tap.calibration_step must be positive")


@dataclass
class ClickSettings:
    """Click stabilization and on-screen feedback settings."""

    cooldown_frames: int = 20
    stable_frames: int = 2
    double_click_interval_frames: int = 8
    feedback_frames: int = 8
    dwell_enabled: bool = False
    dwell_time_ms: int = 600
    dwell_radius_px: int = 12

    def validate(self) -> None:
        """Validate click timing settings."""
        for name in (
            "cooldown_frames",
            "stable_frames",
            "double_click_interval_frames",
            "feedback_frames",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"clicks.{name} cannot be negative")
        if self.stable_frames < 1:
            raise ConfigError("clicks.stable_frames must be at least 1")
        if self.dwell_time_ms < 100:
            raise ConfigError("clicks.dwell_time_ms must be at least 100")
        if self.dwell_radius_px < 2:
            raise ConfigError("clicks.dwell_radius_px must be at least 2")


@dataclass
class DetectionSettings:
    """MediaPipe HandLandmarker confidence and hand-count settings."""

    num_hands: int = 1
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55

    def validate(self) -> None:
        """Validate detection settings."""
        if self.num_hands < 1:
            raise ConfigError("detection.num_hands must be at least 1")
        for name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ConfigError(f"detection.{name} must be between 0 and 1")


@dataclass
class PerformanceSettings:
    """Capture/display throughput and MediaPipe acceleration settings."""

    process_every_n_frames: int = 1
    roi_percentage: float = 0.7
    use_async_processing: bool = True
    use_gpu: bool = False
    async_queue_size: int = 1
    target_fps: float = 60.0

    def validate(self) -> None:
        """Validate performance settings."""
        if self.process_every_n_frames < 1:
            raise ConfigError("performance.process_every_n_frames must be at least 1")
        if not math.isfinite(float(self.roi_percentage)):
            raise ConfigError("performance.roi_percentage must be finite")
        if not 0.0 < self.roi_percentage <= 1.0:
            raise ConfigError("performance.roi_percentage must be in (0, 1]")
        if self.async_queue_size < 1:
            raise ConfigError("performance.async_queue_size must be at least 1")
        if not math.isfinite(float(self.target_fps)):
            raise ConfigError("performance.target_fps must be finite")
        if not 5.0 <= self.target_fps <= 240.0:
            raise ConfigError("performance.target_fps must be between 5 and 240")


@dataclass
class DisplaySettings:
    """On-screen display options."""

    window_name: str = "Virtual Mouse"
    show_landmarks: bool = True
    show_instructions: bool = True
    show_fps: bool = True

    def validate(self) -> None:
        """Validate display settings."""
        if not self.window_name.strip():
            raise ConfigError("display.window_name cannot be empty")


@dataclass
class ModelSettings:
    """Local model path and remote model download URL."""

    path: str = "models/hand_landmarker.task"
    url: str = DEFAULT_MODEL_URL
    download_timeout_seconds: float = 60.0
    expected_sha256: str = ""
    require_https: bool = True

    def validate(self) -> None:
        """Validate model download settings."""
        if not self.path.strip():
            raise ConfigError("model.path cannot be empty")
        if not self.url.startswith(("http://", "https://")):
            raise ConfigError("model.url must be an HTTP or HTTPS URL")
        if self.require_https and self.url.startswith("http://"):
            raise ConfigError("model.url must use HTTPS when require_https is true")
        if not math.isfinite(float(self.download_timeout_seconds)):
            raise ConfigError("model.download_timeout_seconds must be finite")
        if self.download_timeout_seconds <= 0.0:
            raise ConfigError("model.download_timeout_seconds must be positive")
        if self.expected_sha256 and len(self.expected_sha256.strip()) != 64:
            raise ConfigError("model.expected_sha256 must be a 64-char hex digest")


@dataclass
class LogSettings:
    """File logging settings."""

    file: str = "virtual_mouse.log"
    max_bytes: int = 1_000_000
    backup_count: int = 3

    def validate(self) -> None:
        """Validate logging settings."""
        if not self.file.strip():
            raise ConfigError("logging.file cannot be empty")
        if self.max_bytes < 10_000:
            raise ConfigError("logging.max_bytes must be at least 10000")
        if self.backup_count < 0:
            raise ConfigError("logging.backup_count cannot be negative")


@dataclass
class AppConfig:
    """Complete application configuration."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    mouse: MouseSettings = field(default_factory=MouseSettings)
    gestures: GestureSettings = field(default_factory=GestureSettings)
    tap: TapSettings = field(default_factory=TapSettings)
    clicks: ClickSettings = field(default_factory=ClickSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    logging: LogSettings = field(default_factory=LogSettings)

    def validate(self) -> "AppConfig":
        """Validate all nested settings and return ``self`` for chaining."""
        self.camera.validate()
        self.mouse.validate()
        self.gestures.validate()
        self.tap.validate()
        self.clicks.validate()
        self.detection.validate()
        self.performance.validate()
        self.display.validate()
        self.model.validate()
        self.logging.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the configuration."""
        return _dataclass_to_dict(self)

    def save(self, path: str | Path) -> None:
        """Write the current configuration to ``path`` atomically.

        Args:
            path: Destination JSON file. Parent directories are created when
                necessary.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load configuration from JSON.

        Missing keys use the corresponding dataclass defaults. Unknown keys
        are ignored with a warning so a newer configuration file can still be
        used by an older application build.

        Args:
            path: JSON configuration path.

        Returns:
            A validated :class:`AppConfig` instance.

        Raises:
            ConfigError: If the file is missing, malformed, or invalid.
        """
        source = Path(path)
        if not source.exists():
            raise ConfigError(f"Configuration file not found: {source}")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Unable to read configuration {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError("The configuration root must be a JSON object")
        normalized_raw = dict(raw)
        mouse_values = normalized_raw.get("mouse")
        if isinstance(mouse_values, Mapping) and "smoothening" in mouse_values:
            mouse_values = dict(mouse_values)
            legacy_value = mouse_values.pop("smoothening")
            mouse_values.setdefault("smoothing_factor", legacy_value)
            normalized_raw["mouse"] = mouse_values
        config = _from_mapping(cls, normalized_raw, source)
        try:
            config.validate()
        except ConfigError:
            raise
        return config


Config = AppConfig

_T = TypeVar("_T")


def _dataclass_to_dict(value: Any) -> Any:
    """Recursively convert dataclasses to plain dictionaries."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _dataclass_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _dataclass_to_dict(item) for key, item in value.items()}
    return value


def _from_mapping(cls: type[_T], raw: Mapping[str, Any], source: Path) -> _T:
    """Build a dataclass from a mapping, applying defaults recursively."""
    defaults = cls()  # type: ignore[call-arg]
    known_names = {item.name for item in fields(defaults)}
    unknown = set(raw) - known_names
    if unknown:
        LOGGER.warning(
            "Ignoring unknown configuration keys in %s: %s",
            source,
            ", ".join(sorted(str(key) for key in unknown)),
        )

    values: dict[str, Any] = {}
    for item in fields(defaults):
        if item.name not in raw:
            continue
        value = raw[item.name]
        default_value = getattr(defaults, item.name)
        if is_dataclass(default_value) and not isinstance(default_value, type):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{item.name} must be a JSON object")
            values[item.name] = _from_mapping(type(default_value), value, source)
        else:
            values[item.name] = value
    return cls(**values)  # type: ignore[call-arg]
