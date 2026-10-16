"""Entry point for the modern Virtual Mouse application."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from config import AppConfig, ConfigError
from gesture_recognizer import (
    Gesture,
    GestureRecognizer,
    GestureResult,
    ScrollDirection,
    TapDetector,
    TapState,
)
from hand_tracker import (
    HandObservation,
    HandTracker,
    LatestFrameProcessor,
    ProcessedFrame,
    ModelDownloadError,
    download_model,
    draw_hand_landmarks,
)
from mouse_controller import ClickAction, ClickDebouncer, MouseController

LOGGER = logging.getLogger("virtual_mouse")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class CameraError(RuntimeError):
    """Raised when a configured webcam cannot be opened or read."""


@dataclass(frozen=True)
class ClickFeedback:
    """A click rendered on the OSD for a limited number of frames."""

    action: ClickAction
    frame_number: int


class FPSCounter:
    """Exponentially smoothed frames-per-second meter."""

    def __init__(self, smoothing: float = 0.1) -> None:
        """Initialize the meter.

        Args:
            smoothing: Weight assigned to the newest frame interval.
        """
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("FPS smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self.value = 0.0
        self._last_time: float | None = None

    def tick(self) -> float:
        """Record a frame and return the updated FPS value."""
        now = time.perf_counter()
        if self._last_time is None:
            instantaneous = 0.0
        else:
            elapsed = max(now - self._last_time, 1e-9)
            instantaneous = 1.0 / elapsed
        self._last_time = now
        if self.value == 0.0:
            self.value = instantaneous
        else:
            self.value = (
                self.smoothing * instantaneous
                + (1.0 - self.smoothing) * self.value
            )
        return self.value


def configure_logging(level_name: str, log_file: str | None = None) -> None:
    """Configure application logging once for command-line execution."""
    from logging.handlers import RotatingFileHandler

    level = getattr(logging, level_name.upper(), logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            handlers.append(file_handler)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not open log file %s: %s", log_file, exc)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Control a desktop mouse with MediaPipe hand gestures."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json (default: project config.json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional local Hand Landmarker .task model path",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Override camera.index without editing config.json",
    )
    parser.add_argument(
        "--calibrate-tap",
        action="store_true",
        help="Show tap telemetry and adjust tap sensitivity without mouse input",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Personal 10s calibration: pinch, fist, peace then auto-save",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run model, gesture, debounce, and mapping checks without a camera",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for automated smoke tests)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not create an OpenCV window; requires --max-frames",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.no_display and args.max_frames is None:
        parser.error("--no-display requires --max-frames")
    return args


def resolve_model_path(
    config: AppConfig, config_path: Path, override: Path | None
) -> Path:
    """Resolve a model path relative to the configuration file."""
    candidate = override or Path(config.model.path)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.expanduser().resolve()


def prepare_model(
    config: AppConfig, config_path: Path, override: Path | None
) -> Path:
    """Ensure the model exists, downloading it atomically when necessary."""
    if config.model.require_https and config.model.url.startswith("http://"):
        raise ModelDownloadError("Model URL must use HTTPS (see model.require_https)")
    model_path = resolve_model_path(config, config_path, override)
    return download_model(
        model_path,
        model_url=config.model.url,
        timeout_seconds=config.model.download_timeout_seconds,
        expected_sha256=config.model.expected_sha256,
    )


def open_camera(config: AppConfig) -> cv2.VideoCapture:
    """Open and configure the configured webcam with reconnect backoff."""
    attempts = max(1, config.camera.reconnect_attempts + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        capture = cv2.VideoCapture(config.camera.index)
        if capture.isOpened():
            # Keep the driver's queue shallow so the displayed frame and mouse target
            # are as close as possible to the current hand pose.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
            capture.set(cv2.CAP_PROP_FPS, config.camera.fps)
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            LOGGER.info(
                "Camera %d opened (%dx%d requested, %.1f FPS reported, attempt %d/%d)",
                config.camera.index,
                actual_width,
                actual_height,
                actual_fps,
                attempt,
                attempts,
            )
            return capture
        capture.release()
        last_error = CameraError(
            f"Could not open camera index {config.camera.index}. "
            "Check camera permissions and camera.index in config.json."
        )
        if attempt < attempts:
            LOGGER.warning("Camera open failed (attempt %d/%d), retrying...", attempt, attempts)
            time.sleep(config.camera.reconnect_backoff_seconds)
    raise last_error if last_error else CameraError("Could not open camera")


def click_action_for(gesture: Gesture) -> ClickAction | None:
    """Map a held gesture to its debounced mouse action.

    Pinch is intentionally excluded: left click is emitted only by the
    stateful index-finger tap detector.
    """
    if gesture is Gesture.FIST:
        return ClickAction.RIGHT_CLICK
    if gesture is Gesture.PEACE:
        return ClickAction.DOUBLE_CLICK
    return None


def gesture_color(gesture: Gesture | None) -> tuple[int, int, int]:
    """Return the BGR accent color used for a gesture."""
    if gesture is Gesture.FIST:
        return (0, 0, 255)
    if gesture is Gesture.PINCH:
        return (0, 200, 255)
    if gesture is Gesture.PEACE:
        return (255, 180, 0)
    if gesture in (Gesture.SCROLL_UP, Gesture.SCROLL_DOWN):
        return (220, 0, 220)
    if gesture is Gesture.POINTING:
        return (0, 255, 0)
    return (0, 200, 255)


def click_color(action: ClickAction) -> tuple[int, int, int]:
    """Return the BGR color used for click feedback."""
    if action is ClickAction.RIGHT_CLICK:
        return (0, 0, 255)
    if action is ClickAction.DOUBLE_CLICK:
        return (255, 180, 0)
    return (0, 255, 0)


def _draw_translucent_panel(
    frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> None:
    """Draw a semi-transparent panel in place."""
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, top_left[0]))
    top = max(0, min(height - 1, top_left[1]))
    right = max(left + 1, min(width, bottom_right[0]))
    bottom = max(top + 1, min(height, bottom_right[1]))
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)


def _draw_status_panel(
    frame: np.ndarray,
    fps: float,
    result: GestureResult | None,
    observation: HandObservation | None,
    frame_number: int,
    debouncer: ClickDebouncer,
    feedback: ClickFeedback | None,
    tap_detector: TapDetector | None = None,
    tap_feedback: bool = False,
    calibration_mode: bool = False,
    processing_mode: str = "",
    show_fps: bool = True,
    paused: bool = False,
    second_result: GestureResult | None = None,
    backend_name: str = "",
    dragging: bool = False,
) -> None:
    """Draw status, FPS, gesture, tap state, and debounce information."""
    height, width = frame.shape[:2]
    scale = 0.48 if width < 900 else 0.58
    line_height = max(20, int(25 * scale / 0.48))
    panel_width = min(width - 16, 560)
    panel_height = min(height - 16, 24 + line_height * 7)
    _draw_translucent_panel(
        frame, (8, 8), (8 + panel_width, 8 + panel_height)
    )
    text_x = 18
    text_y = 8 + line_height
    status = result.label if result is not None else Gesture.NO_HAND.value
    if paused:
        status = "PAUSED (2nd fist)"
    elif dragging:
        status += " + DRAG"
    color = (0, 0, 255) if paused else gesture_color(result.gesture if result is not None else None)
    cv2.putText(
        frame,
        f"Gesture: {status}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )
    text_y += line_height
    hand_text = (
        observation.handedness if observation is not None else "waiting for hand"
    )
    mode_text = f" | {processing_mode}" if processing_mode else ""
    cv2.putText(
        frame,
        f"Hand: {hand_text} | Frame: {frame_number}{mode_text}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    text_y += line_height
    if show_fps:
        cv2.putText(
            frame,
            (
                f"FPS: {fps:5.1f} | Click cooldown: "
                f"{debouncer.frames_until_ready(frame_number)}f"
            ),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    text_y += line_height

    if tap_detector is not None:
        tap_line = (
            f"Tap: {tap_detector.state.value} | "
            f"v={tap_detector.last_tap_velocity:.3f} "
            f"drop={tap_detector.last_displacement:.3f}"
        )
        tap_color = (0, 255, 0) if tap_detector.is_armed else (0, 200, 255)
        cv2.putText(
            frame,
            tap_line,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            tap_color,
            1,
            cv2.LINE_AA,
        )
    text_y += line_height
    second_line = f"2nd: {second_result.gesture.value}" if second_result is not None else "2nd: -"
    backend_line = f" | {backend_name}" if backend_name else ""
    cv2.putText(
        frame,
        f"{second_line}{backend_line}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    text_y += line_height
    if calibration_mode:
        cv2.putText(
            frame,
            "CALIBRATION: [ ] velocity | - = tip drop | q quit",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    elif tap_feedback:
        cv2.putText(
            frame,
            "TAP!",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    text_y += line_height
    if feedback is not None:
        cv2.putText(
            frame,
            f"CLICK: {feedback.action.value.upper()}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            click_color(feedback.action),
            2,
            cv2.LINE_AA,
        )


def _draw_instructions(
    frame: np.ndarray, calibration_mode: bool = False
) -> None:
    """Draw concise controls and gesture instructions."""
    height, width = frame.shape[:2]
    if calibration_mode:
        lines = (
            "Tap calibration: extend index, then quickly curl and re-extend",
            "[ ] velocity | - = tip drop | r reload | s save | q quit",
        )
    else:
        lines = (
            "Index: move | Tap index: left click | Fist: right | Peace: double",
            "Pinch: no click | scroll: thumb up/down | [ ] smoothing | q quit",
        )
    scale = 0.42 if width < 900 else 0.50
    text_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        for line in lines
    )
    panel_width = min(width - 12, text_width + 24)
    panel_height = 58
    top = max(0, height - panel_height - 8)
    _draw_translucent_panel(frame, (6, top), (6 + panel_width, top + panel_height))
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, top + 23 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def _draw_feedback_marker(
    frame: np.ndarray,
    observation: HandObservation | None,
    feedback: ClickFeedback | None,
    current_frame: int,
    feedback_frames: int,
) -> None:
    """Draw a colored expanding ring after a click is registered."""
    if observation is None or feedback is None:
        return
    age = current_frame - feedback.frame_number
    if age < 0 or age >= max(1, feedback_frames):
        return
    if len(observation.landmarks) <= 8:
        return
    tip = observation.landmarks[8]
    if tip.x is None or tip.y is None:
        return
    height, width = frame.shape[:2]
    point = (
        max(0, min(width - 1, int(tip.x * (width - 1)))),
        max(0, min(height - 1, int(tip.y * (height - 1)))),
    )
    progress = age / max(1, feedback_frames - 1)
    radius = int(10 + 24 * progress)
    color = click_color(feedback.action)
    cv2.circle(frame, point, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)


def _synthetic_landmarks(pose: str) -> list[NormalizedLandmark]:
    """Create small deterministic landmark sets for the self-test."""
    if pose == "fist":
        coordinates = [
            (0.50, 0.86),
            (0.40, 0.78), (0.43, 0.76), (0.45, 0.75), (0.46, 0.74),
            (0.43, 0.72), (0.44, 0.76), (0.45, 0.78), (0.46, 0.79),
            (0.50, 0.71), (0.50, 0.75), (0.50, 0.77), (0.50, 0.78),
            (0.57, 0.72), (0.56, 0.76), (0.55, 0.78), (0.54, 0.79),
            (0.63, 0.74), (0.61, 0.77), (0.60, 0.79), (0.59, 0.80),
        ]
    elif pose == "peace":
        coordinates = [
            (0.50, 0.86),
            (0.39, 0.78), (0.34, 0.70), (0.31, 0.62), (0.29, 0.55),
            (0.45, 0.66), (0.45, 0.50), (0.45, 0.36), (0.45, 0.22),
            (0.55, 0.66), (0.55, 0.49), (0.55, 0.34), (0.55, 0.20),
            (0.61, 0.69), (0.61, 0.62), (0.64, 0.60), (0.65, 0.57),
            (0.67, 0.72), (0.67, 0.66), (0.70, 0.63), (0.71, 0.60),
        ]
    elif pose == "pinch":
        coordinates = [
            (0.50, 0.86),
            (0.40, 0.78), (0.39, 0.66), (0.42, 0.43), (0.45, 0.25),
            (0.45, 0.66), (0.45, 0.50), (0.45, 0.36), (0.45, 0.24),
            (0.52, 0.66), (0.52, 0.59), (0.52, 0.62), (0.52, 0.63),
            (0.59, 0.68), (0.59, 0.61), (0.59, 0.64), (0.59, 0.65),
            (0.66, 0.71), (0.66, 0.65), (0.66, 0.68), (0.66, 0.69),
        ]
    else:  # pointing
        coordinates = [
            (0.50, 0.86),
            (0.39, 0.78), (0.34, 0.70), (0.31, 0.62), (0.29, 0.55),
            (0.45, 0.66), (0.45, 0.50), (0.45, 0.36), (0.45, 0.22),
            (0.52, 0.66), (0.52, 0.59), (0.52, 0.62), (0.52, 0.63),
            (0.59, 0.68), (0.59, 0.61), (0.59, 0.64), (0.59, 0.65),
            (0.66, 0.71), (0.66, 0.65), (0.66, 0.68), (0.66, 0.69),
        ]
    return [
        NormalizedLandmark(x=x, y=y, z=0.0)
        for x, y in coordinates
    ]


def _tap_landmarks(index_tip_y: float, index_pip_y: float = 0.5) -> list[NormalizedLandmark]:
    """Create a minimal landmark set for tap-detector self-tests."""
    landmarks = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[6] = NormalizedLandmark(x=0.45, y=index_pip_y, z=0.0)
    landmarks[8] = NormalizedLandmark(x=0.45, y=index_tip_y, z=0.0)
    return landmarks


def run_self_test(config_path: Path, model_override: Path | None = None) -> int:
    """Run deterministic checks that do not require a physical webcam."""
    config = AppConfig.load(config_path)
    model_path = prepare_model(config, config_path, model_override)
    recognizer = GestureRecognizer(config.gestures, config.tap)

    cases = {
        "pointing": Gesture.POINTING,
        "pinch": Gesture.PINCH,
        "fist": Gesture.FIST,
        "peace": Gesture.PEACE,
    }
    for pose, expected in cases.items():
        actual = recognizer.recognize(_synthetic_landmarks(pose)).gesture
        if actual is not expected:
            raise AssertionError(
                f"{pose}: expected {expected.value}, got {actual.value}"
            )
        LOGGER.info("Gesture self-test passed: %s -> %s", pose, actual.value)

    if click_action_for(Gesture.PINCH) is not None:
        raise AssertionError("pinch must not emit a left click")
    tap_detector = recognizer.tap_detector
    tap_frames = [
        (0.20, False),
        (0.20, False),
        (0.55, False),
        (0.70, False),
        (0.85, True),
        (0.85, False),
        (0.20, False),
        (0.20, False),
        (0.20, False),
        (0.55, False),
        (0.70, False),
        (0.85, True),
    ]
    tap_events = [
        recognizer.detect_tap_gesture(
            _tap_landmarks(tip_y), frame_number=frame_number
        )
        for frame_number, (tip_y, _) in enumerate(tap_frames)
    ]
    if tap_events != [expected for _, expected in tap_frames]:
        raise AssertionError(f"tap state machine failed: {tap_events}")
    if tap_detector.state is not TapState.TAPPED or tap_detector.is_armed:
        raise AssertionError("tap detector did not enter the re-arm lockout")
    rearm_frame = len(tap_frames)
    recognizer.detect_tap_gesture(
        _tap_landmarks(0.20), frame_number=rearm_frame
    )
    recognizer.detect_tap_gesture(
        _tap_landmarks(0.20), frame_number=rearm_frame + 1
    )
    recognizer.detect_tap_gesture(
        _tap_landmarks(0.20), frame_number=rearm_frame + 2
    )
    if tap_detector.state is not TapState.EXTENDED or not tap_detector.is_armed:
        raise AssertionError("tap detector did not re-arm after extension")
    slow_detector = TapDetector(config.tap)
    slow_events = [
        slow_detector.update(tip_y, 0.5, frame_number)
        for frame_number, tip_y in enumerate((0.20, 0.20, 0.22, 0.24, 0.26, 0.20, 0.20))
    ]
    if any(slow_events):
        raise AssertionError(f"slow curl produced a false tap: {slow_events}")
    LOGGER.info("Tap detector self-test passed (fast tap, re-arm, slow-curl rejection)")

    debouncer = ClickDebouncer(
        cooldown_frames=config.clicks.cooldown_frames,
        stable_frames=config.clicks.stable_frames,
    )
    if debouncer.update(ClickAction.LEFT_CLICK, 0) is not None:
        raise AssertionError("debouncer fired before the gesture was stable")
    if debouncer.update(ClickAction.LEFT_CLICK, 1) is not ClickAction.LEFT_CLICK:
        raise AssertionError("debouncer did not fire on a stable gesture")
    if debouncer.update(ClickAction.LEFT_CLICK, 2) is not None:
        raise AssertionError("debouncer repeated a held gesture")
    debouncer.update(None, 3)
    if debouncer.update(ClickAction.LEFT_CLICK, 4) is not None:
        raise AssertionError("debouncer ignored the cooldown")
    edge_debouncer = ClickDebouncer(20, 2)
    if edge_debouncer.emit_edge(ClickAction.LEFT_CLICK, 0) is not ClickAction.LEFT_CLICK:
        raise AssertionError("edge debouncer did not emit a tap")
    if edge_debouncer.emit_edge(ClickAction.LEFT_CLICK, 1) is not None:
        raise AssertionError("edge debouncer ignored cooldown")
    LOGGER.info("Click debounce self-test passed")

    controller = MouseController(config.mouse, screen_size=(800, 600))
    mapped = controller.map_to_screen(0.5, 0.5)
    if not (399 <= mapped[0] <= 401 and 299 <= mapped[1] <= 301):
        raise AssertionError(f"screen mapping self-test failed: {mapped}")
    LOGGER.info("Screen mapping self-test passed: %s", mapped)

    from mouse_controller import DwellClickDetector, OneEuroFilter

    euro = OneEuroFilter(1.0, 0.05, 1.0)
    first = euro.filter(100.0, 0.0)
    second = euro.filter(100.0, 0.016)
    if abs(first - 100.0) > 1e-6 or abs(second - 100.0) > 1e-3:
        raise AssertionError(f"OneEuro steady-state failed: {first}, {second}")
    LOGGER.info("OneEuro filter self-test passed")

    dwell = DwellClickDetector(600, 12)
    from mouse_controller import ScreenPoint

    if dwell.update(ScreenPoint(100, 100), now=0.0):
        raise AssertionError("dwell fired too early")
    if not dwell.update(ScreenPoint(101, 101), now=0.7):
        raise AssertionError("dwell did not fire after 700ms hover")
    if dwell.update(ScreenPoint(102, 102), now=0.8):
        raise AssertionError("dwell re-fired without reset")
    LOGGER.info("Dwell detector self-test passed")

    pinch_landmarks = _synthetic_landmarks("pinch")
    drag_states = [recognizer.detect_drag(pinch_landmarks) for _ in range(config.gestures.drag_hold_frames + 2)]
    if not drag_states[-1][0] or not any(s[1] for s in drag_states):
        raise AssertionError(f"pinch-hold drag did not start: {drag_states[-1]}")
    open_landmarks = _synthetic_landmarks("pointing")
    release_states = [recognizer.detect_drag(open_landmarks) for _ in range(config.gestures.drag_release_frames + 2)]
    if release_states[-1][0] or not any(s[2] for s in release_states):
        raise AssertionError(f"drag did not release: {release_states[-1]}")
    LOGGER.info("Pinch-hold drag self-test passed")

    with HandTracker(
        model_path,
        num_hands=config.detection.num_hands,
        min_detection_confidence=config.detection.min_hand_detection_confidence,
        min_presence_confidence=config.detection.min_hand_presence_confidence,
        min_tracking_confidence=config.detection.min_tracking_confidence,
        use_gpu=config.performance.use_gpu,
    ) as tracker:
        blank_frame = np.zeros(
            (config.camera.height, config.camera.width, 3), dtype=np.uint8
        )
        for timestamp in (1, 2, 3):
            tracker.detect(
                blank_frame,
                timestamp,
                roi_percentage=config.performance.roi_percentage,
            )
    LOGGER.info("MediaPipe VIDEO-mode self-test passed")
    LOGGER.info("Self-test completed successfully")
    return 0


def run_calibration(args: argparse.Namespace) -> int:
    """Personal calibration: sample pinch/fist/peace and save thresholds."""
    from gesture_recognizer import fingertip_wrist_distances, landmark_distance

    config_path = args.config.expanduser().resolve()
    config = AppConfig.load(config_path)
    model_path = prepare_model(config, config_path, args.model)
    capture = open_camera(config)
    tracker = HandTracker(
        model_path,
        num_hands=1,
        min_detection_confidence=config.detection.min_hand_detection_confidence,
        min_presence_confidence=config.detection.min_hand_presence_confidence,
        min_tracking_confidence=config.detection.min_tracking_confidence,
    )
    poses = [
        ("PINCH thumb+index tightly (3s)", 90),
        ("FIST all fingers closed (3s)", 90),
        ("PEACE index+middle apart (3s)", 90),
    ]
    samples: dict[str, list[float]] = {"pinch": [], "fist": [], "peace": []}
    try:
        ts = 0
        for label, frames in poses:
            LOGGER.info("Calibrate: %s", label)
            for _ in range(frames):
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                if config.camera.flip:
                    frame = cv2.flip(frame, 1)
                ts += 1
                obs = tracker.detect(frame, ts, roi_percentage=config.performance.roi_percentage)
                if obs is None:
                    continue
                if "PINCH" in label:
                    samples["pinch"].append(landmark_distance(obs.landmarks[4], obs.landmarks[8]))
                elif "FIST" in label:
                    d = fingertip_wrist_distances(obs.landmarks)
                    samples["fist"].append(max(d.values()))
                else:
                    samples["peace"].append(landmark_distance(obs.landmarks[8], obs.landmarks[12]))
                if not args.no_display:
                    cv2.putText(frame, f"Calibrate: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow(config.display.window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return 0
        if samples["pinch"]:
            config.gestures.pinch_threshold = round(max(samples["pinch"]) * 1.25 + 0.005, 4)
        if samples["fist"]:
            config.gestures.fist_tip_wrist_threshold = round(max(samples["fist"]) * 1.2 + 0.02, 4)
        if samples["peace"]:
            config.gestures.peace_min_separation = round(min(samples["peace"]) * 0.8, 4)
        config.validate()
        config.save(config_path)
        LOGGER.info(
            "Calibration saved: pinch=%.4f fist=%.4f peace=%.4f",
            config.gestures.pinch_threshold,
            config.gestures.fist_tip_wrist_threshold,
            config.gestures.peace_min_separation,
        )
        return 0
    finally:
        capture.release()
        tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def run_application(args: argparse.Namespace) -> int:
    """Run the camera, tracking, gesture, and mouse loop."""
    config_path = args.config.expanduser().resolve()
    config = AppConfig.load(config_path)
    if args.camera is not None:
        config.camera.index = args.camera
        config.camera.validate()
    model_path = prepare_model(config, config_path, args.model)

    tracker = HandTracker(
        model_path,
        num_hands=config.detection.num_hands,
        min_detection_confidence=config.detection.min_hand_detection_confidence,
        min_presence_confidence=config.detection.min_hand_presence_confidence,
        min_tracking_confidence=config.detection.min_tracking_confidence,
        use_gpu=config.performance.use_gpu,
    )
    recognizer = GestureRecognizer(config.gestures, config.tap)
    processor: LatestFrameProcessor | None = None
    mouse: MouseController | None = None
    try:
        if not args.calibrate_tap:
            mouse = MouseController(config.mouse)
        debouncer = ClickDebouncer(
            cooldown_frames=config.clicks.cooldown_frames,
            stable_frames=config.clicks.stable_frames,
        )
        if config.performance.use_async_processing:
            processor = LatestFrameProcessor(
                tracker,
                roi_percentage=config.performance.roi_percentage,
                queue_size=config.performance.async_queue_size,
            )
        capture = open_camera(config)
    except Exception:
        if processor is not None:
            processor.close()
        tracker.close()
        raise

    frame_number = 0
    failed_reads = 0
    reconnects = 0
    feedback: ClickFeedback | None = None
    tap_feedback_frame: int | None = None
    tap_feedback_until = -1
    tap_conflict_until = -1
    last_scroll_frame = -10_000
    last_observation: HandObservation | None = None
    last_result: GestureResult | None = None
    second_observation: HandObservation | None = None
    second_result: GestureResult | None = None
    paused = False
    last_second_action_frame = -10_000
    last_detection_frame = -1
    fps_counter = FPSCounter()
    processing_mode = "async" if processor is not None else "sync"
    dwell_detector = None
    if config.clicks.dwell_enabled:
        from mouse_controller import DwellClickDetector

        dwell_detector = DwellClickDetector(
            config.clicks.dwell_time_ms, config.clicks.dwell_radius_px
        )

    try:
        while True:
            frame_start = time.perf_counter()
            success, frame = capture.read()
            if not success or frame is None:
                failed_reads += 1
                if failed_reads >= 5:
                    if reconnects < config.camera.reconnect_attempts:
                        reconnects += 1
                        LOGGER.warning(
                            "Camera lost %d frames, reconnecting %d/%d...",
                            failed_reads,
                            reconnects,
                            config.camera.reconnect_attempts,
                        )
                        try:
                            capture.release()
                        except Exception:
                            pass
                        time.sleep(config.camera.reconnect_backoff_seconds)
                        try:
                            capture = open_camera(config)
                        except CameraError as exc:
                            LOGGER.error("Camera reconnect failed: %s", exc)
                            raise
                        failed_reads = 0
                        continue
                    raise CameraError(
                        "The webcam stopped returning frames after five "
                        "consecutive reads"
                    )
                LOGGER.warning("Dropped webcam frame (%d/5)", failed_reads)
                time.sleep(0.02)
                continue
            failed_reads = 0
            reconnects = 0
            if config.camera.flip:
                frame = cv2.flip(frame, 1)

            should_process = (
                frame_number % config.performance.process_every_n_frames == 0
            )
            completed: ProcessedFrame | None = None
            if processor is not None:
                if should_process:
                    timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                    processor.submit(frame, frame_number, timestamp_ms)
                completed = processor.take_latest()
            elif should_process:
                timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                observations = tracker.detect_all(
                    frame,
                    timestamp_ms,
                    roi_percentage=config.performance.roi_percentage,
                )
                observation = observations[0] if observations else None
                completed = ProcessedFrame(frame_number, timestamp_ms, observation, None, observations)

            tap_event = False
            drag_started = False
            drag_ended = False
            dragging = mouse.is_dragging if mouse is not None else False
            if completed is not None:
                processed = completed
                if processed.error is not None:
                    LOGGER.error("Hand processing error: %s", processed.error)
                    recognizer.reset_tap()
                    if mouse is not None and mouse.is_dragging:
                        mouse.set_dragging(False)
                    if dwell_detector is not None:
                        dwell_detector.reset()
                    last_observation = None
                    last_result = None
                else:
                    last_detection_frame = processed.frame_number
                    observation = processed.observation
                    if observation is None:
                        recognizer.reset_tap()
                        if mouse is not None and mouse.is_dragging:
                            mouse.set_dragging(False)
                        if dwell_detector is not None:
                            dwell_detector.reset()
                        last_observation = None
                        last_result = None
                        second_observation = None
                        second_result = None
                        paused = False
                    else:
                        last_observation = observation
                        last_result = recognizer.recognize(observation.landmarks)
                        dragging, drag_started, drag_ended = recognizer.detect_drag(
                            observation.landmarks
                        )
                        if not args.calibrate_tap and mouse is not None:
                            if drag_started:
                                mouse.set_dragging(True)
                            elif drag_ended:
                                mouse.set_dragging(False)
                        suppress_tap = last_result.gesture in (
                            Gesture.PINCH,
                            Gesture.PEACE,
                            Gesture.SCROLL_UP,
                            Gesture.SCROLL_DOWN,
                        ) or dragging
                        if recognizer.detect_tap_gesture(
                            observation.landmarks,
                            frame_number=processed.frame_number,
                            suppress=suppress_tap,
                        ):
                            tap_event = True
                            tap_feedback_frame = frame_number
                            tap_feedback_until = (
                                frame_number + config.clicks.feedback_frames
                            )
                            tap_conflict_until = (
                                frame_number + config.tap.conflict_lock_frames
                            )
                        # Second hand: pause + shortcuts (only when 2+ hands).
                        all_obs = processed.observations or ((observation,) if observation else ())
                        if len(all_obs) > 1 and all_obs[1] is not None:
                            second_observation = all_obs[1]
                            second_result = recognizer.recognize(second_observation.landmarks)
                            if not args.calibrate_tap and mouse is not None:
                                if second_result.gesture is Gesture.FIST:
                                    if not paused:
                                        paused = True
                                        if mouse.is_dragging:
                                            mouse.set_dragging(False)
                                        LOGGER.info("Second-hand pause ON at frame %d", frame_number)
                                elif second_result.gesture in (Gesture.POINTING, Gesture.UNKNOWN):
                                    if paused:
                                        paused = False
                                        LOGGER.info("Second-hand pause OFF at frame %d", frame_number)
                                if (
                                    frame_number - last_second_action_frame
                                    >= config.clicks.cooldown_frames
                                ):
                                    if second_result.gesture is Gesture.PINCH:
                                        if mouse.perform_click(ClickAction.MIDDLE_CLICK):
                                            last_second_action_frame = frame_number
                                            feedback = ClickFeedback(ClickAction.MIDDLE_CLICK, frame_number)
                                            LOGGER.info("Second-hand middle click at frame %d", frame_number)
                                    elif second_result.gesture is Gesture.PEACE:
                                        if mouse.hotkey("ctrl", "c"):
                                            last_second_action_frame = frame_number
                                            LOGGER.info("Second-hand copy (ctrl+c) at frame %d", frame_number)
                        else:
                            second_observation = None
                            second_result = None

            observation = last_observation
            result = last_result
            tap_event_active = (
                tap_feedback_frame is not None
                and frame_number < tap_feedback_until
            )

            if (
                not args.calibrate_tap
                and not paused
                and mouse is not None
                and result is not None
                and (result.gesture is Gesture.POINTING or dragging)
            ):
                index_tip = observation.landmarks[8] if observation else None
                if index_tip is not None and index_tip.x is not None and index_tip.y is not None:
                    cursor_point = mouse.move_to_normalized(index_tip.x, index_tip.y)
                    if (
                        dwell_detector is not None
                        and not dragging
                        and result.gesture is Gesture.POINTING
                        and dwell_detector.update(cursor_point)
                    ):
                        tap_event = True
                        tap_feedback_frame = frame_number
                        tap_feedback_until = frame_number + config.clicks.feedback_frames
                        tap_conflict_until = frame_number + config.tap.conflict_lock_frames

            scroll_dir = None
            if not args.calibrate_tap and last_observation is not None:
                scroll_dir = recognizer.scroll_with_hysteresis(last_observation.landmarks)
            if (
                not args.calibrate_tap
                and not paused
                and mouse is not None
                and scroll_dir is not None
                and frame_number - last_scroll_frame
                >= config.gestures.scroll_cooldown_frames
            ):
                if mouse.scroll(scroll_dir.value, config.gestures.scroll_amount):
                    last_scroll_frame = frame_number

            if not args.calibrate_tap and not paused:
                if tap_event:
                    # A detected tap owns the click path and briefly suppresses
                    # a simultaneous fist/peace classification.
                    debouncer.reset()
                    emitted_click = debouncer.emit_edge(
                        ClickAction.LEFT_CLICK, frame_number
                    )
                else:
                    click_action = (
                        click_action_for(result.gesture)
                        if result is not None
                        and frame_number >= tap_conflict_until
                        else None
                    )
                    emitted_click = debouncer.update(click_action, frame_number)
            else:
                emitted_click = None

            if (
                emitted_click is not None
                and mouse is not None
                and mouse.perform_click(emitted_click)
            ):
                feedback = ClickFeedback(emitted_click, frame_number)
                LOGGER.info(
                    "Registered %s at frame %d", emitted_click.value, frame_number
                )
            if (
                feedback is not None
                and frame_number - feedback.frame_number
                >= config.clicks.feedback_frames
            ):
                feedback = None
            if tap_feedback_frame is not None and frame_number >= tap_feedback_until:
                tap_feedback_frame = None

            if config.display.show_landmarks and observation is not None:
                draw_hand_landmarks(
                    frame,
                    observation.landmarks,
                    landmark_color=gesture_color(result.gesture if result else None),
                )
            if config.display.show_landmarks and second_observation is not None:
                draw_hand_landmarks(
                    frame,
                    second_observation.landmarks,
                    landmark_color=(255, 255, 255),
                )
            tap_marker_feedback = feedback
            if tap_event_active and tap_feedback_frame is not None:
                tap_marker_feedback = ClickFeedback(
                    ClickAction.LEFT_CLICK, tap_feedback_frame
                )
            _draw_feedback_marker(
                frame,
                observation,
                tap_marker_feedback,
                frame_number,
                config.clicks.feedback_frames,
            )
            _draw_status_panel(
                frame,
                fps_counter.tick(),
                result,
                observation,
                frame_number,
                debouncer,
                feedback,
                tap_detector=recognizer.tap_detector,
                tap_feedback=tap_event_active,
                calibration_mode=args.calibrate_tap,
                processing_mode=processing_mode,
                show_fps=config.display.show_fps,
                paused=paused,
                second_result=second_result,
                backend_name=mouse.backend_name if mouse is not None else "",
                dragging=dragging,
            )
            if config.display.show_instructions:
                _draw_instructions(frame, args.calibrate_tap)

            if not args.no_display:
                try:
                    cv2.imshow(config.display.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV could not create or update the preview window"
                    ) from exc
                if key in (ord("q"), 27):
                    LOGGER.info("Quit requested")
                    break
                if key == ord("["):
                    if args.calibrate_tap:
                        config.tap.velocity_threshold = max(
                            config.tap.calibration_step,
                            config.tap.velocity_threshold
                            - config.tap.calibration_step,
                        )
                        recognizer.tap_detector.update_settings(config.tap)
                        LOGGER.info(
                            "Tap velocity threshold: %.3f",
                            config.tap.velocity_threshold,
                        )
                    elif mouse is not None:
                        factor = max(1.0, config.mouse.smoothing_factor - 0.5)
                        config.mouse.smoothing_factor = factor
                        mouse.set_smoothing_factor(factor)
                        LOGGER.info("Smoothing factor: %.1f", factor)
                elif key == ord("]"):
                    if args.calibrate_tap:
                        config.tap.velocity_threshold += config.tap.calibration_step
                        recognizer.tap_detector.update_settings(config.tap)
                        LOGGER.info(
                            "Tap velocity threshold: %.3f",
                            config.tap.velocity_threshold,
                        )
                    elif mouse is not None:
                        factor = config.mouse.smoothing_factor + 0.5
                        config.mouse.smoothing_factor = factor
                        mouse.set_smoothing_factor(factor)
                        LOGGER.info("Smoothing factor: %.1f", factor)
                elif key in (ord("-"), ord("=")) and args.calibrate_tap:
                    if key == ord("="):
                        config.tap.min_tip_drop += config.tap.calibration_step
                    else:
                        config.tap.min_tip_drop = max(
                            0.0,
                            config.tap.min_tip_drop - config.tap.calibration_step,
                        )
                    recognizer.tap_detector.update_settings(config.tap)
                    LOGGER.info(
                        "Tap minimum tip drop: %.3f", config.tap.min_tip_drop
                    )
                elif key == ord("r"):
                    try:
                        reloaded = AppConfig.load(config_path)
                        if args.camera is not None:
                            reloaded.camera.index = args.camera
                        previous_model_path = model_path
                        config = reloaded
                        if mouse is not None:
                            mouse.update_settings(config.mouse)
                        recognizer.update_settings(config.gestures, config.tap)
                        debouncer = ClickDebouncer(
                            cooldown_frames=config.clicks.cooldown_frames,
                            stable_frames=config.clicks.stable_frames,
                        )
                        LOGGER.info("Reloaded configuration from %s", config_path)
                        if (
                            resolve_model_path(config, config_path, args.model)
                            != previous_model_path
                        ):
                            LOGGER.warning("Model changes take effect after restart")
                        if processor is not None:
                            LOGGER.warning(
                                "ROI/frame-cadence changes take effect after restart"
                            )
                    except ConfigError as exc:
                        LOGGER.error("Could not reload configuration: %s", exc)
                elif key == ord("s"):
                    try:
                        config.save(config_path)
                        LOGGER.info("Saved configuration to %s", config_path)
                    except OSError as exc:
                        LOGGER.error("Could not save configuration: %s", exc)
                elif key == ord("h"):
                    config.display.show_instructions = (
                        not config.display.show_instructions
                    )
                    LOGGER.info(
                        "Instructions %s",
                        "shown" if config.display.show_instructions else "hidden",
                    )
                elif key == ord("d"):
                    config.display.show_landmarks = (
                        not config.display.show_landmarks
                    )
                    LOGGER.info(
                        "Landmarks %s",
                        "shown" if config.display.show_landmarks else "hidden",
                    )

            frame_number += 1
            if frame_number % 100 == 0:
                LOGGER.info(
                    "FPS checkpoint: %.1f (frame %d, last detection %d)",
                    fps_counter.value,
                    frame_number,
                    last_detection_frame,
                )
            if args.max_frames is not None and frame_number >= args.max_frames:
                LOGGER.info("Reached configured maximum of %d frames", frame_number)
                break
            elapsed = time.perf_counter() - frame_start
            min_frame_time = 1.0 / max(5.0, config.performance.target_fps)
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)
    finally:
        try:
            if mouse is not None and mouse.is_dragging:
                mouse.set_dragging(False)
        except Exception:
            pass
        if processor is not None:
            processor.close()
        capture.release()
        tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        LOGGER.info(
            "Virtual Mouse stopped after %d frames (average %.1f FPS)",
            frame_number,
            fps_counter.value,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point with user-friendly error handling."""
    args = parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        if config_path.exists():
            try:
                early_config = AppConfig.load(config_path)
                configure_logging(args.log_level, early_config.logging.file)
            except ConfigError:
                configure_logging(args.log_level)
        else:
            configure_logging(args.log_level)
    except Exception:
        configure_logging(args.log_level)
    try:
        if args.self_test:
            return run_self_test(args.config.expanduser().resolve(), args.model)
        if args.calibrate:
            return run_calibration(args)
        return run_application(args)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 0
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2
    except (CameraError, ModelDownloadError, FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("Virtual Mouse could not start: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected Virtual Mouse failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
