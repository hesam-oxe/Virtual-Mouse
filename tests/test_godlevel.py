"""God-level regression tests for Virtual Mouse."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from gesture_recognizer import Gesture, GestureRecognizer
from main import _synthetic_landmarks
from mouse_controller import ClickDebouncer, ClickAction, DwellClickDetector, MouseController, OneEuroFilter


def test_gestures():
    r = GestureRecognizer()
    assert r.recognize(_synthetic_landmarks("pointing")).gesture is Gesture.POINTING
    assert r.recognize(_synthetic_landmarks("pinch")).gesture is Gesture.PINCH
    assert r.recognize(_synthetic_landmarks("fist")).gesture is Gesture.FIST
    assert r.recognize(_synthetic_landmarks("peace")).gesture is Gesture.PEACE


def test_drag_start_release():
    r = GestureRecognizer()
    pinch = _synthetic_landmarks("pinch")
    for _ in range(r.settings.drag_hold_frames + 2):
        dragging, started, _ = r.detect_drag(pinch)
    assert dragging and started or dragging
    pointing = _synthetic_landmarks("pointing")
    ended = False
    for _ in range(r.settings.drag_release_frames + 2):
        dragging, _, ended_flag = r.detect_drag(pointing)
        ended = ended or ended_flag
    assert not dragging and ended


def test_scroll_hysteresis_holds():
    r = GestureRecognizer()
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

    def lm(x, y):
        return NormalizedLandmark(x=x, y=y, z=0.0)

    # All fingers curled, thumb up -> UP
    base = [lm(0.5, 0.5) for _ in range(21)]
    base[0] = lm(0.5, 0.6)  # wrist
    base[4] = lm(0.5, 0.3)  # thumb up
    for tip in (8, 12, 16, 20):
        base[tip] = lm(0.52, 0.58)
    assert r.scroll_with_hysteresis(base) is not None


def test_one_euro_steady():
    f = OneEuroFilter(1.0, 0.05, 1.0)
    assert abs(f.filter(10.0, 0.0) - 10.0) < 1e-9
    assert abs(f.filter(10.0, 0.016) - 10.0) < 1e-3


def test_debouncer_cooldown():
    d = ClickDebouncer(20, 2)
    assert d.update(ClickAction.LEFT_CLICK, 0) is None
    assert d.update(ClickAction.LEFT_CLICK, 1) is ClickAction.LEFT_CLICK
    assert d.update(ClickAction.LEFT_CLICK, 2) is None


def test_config_defaults():
    c = AppConfig.load(Path(__file__).resolve().parent.parent / "config.json")
    c.validate()
    assert c.mouse.smoothing_backend in ("exponential", "one_euro")
    assert c.gestures.drag_hold_frames >= 1
    assert c.performance.target_fps >= 5.0
