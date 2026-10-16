"""Gesture classification for MediaPipe hand landmarks.

The classifier deliberately uses landmark geometry rather than screen-space
pixel thresholds.  This keeps thresholds meaningful when the camera resolution
changes and makes each gesture independently testable.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from config import GestureSettings, TapSettings

LOGGER = logging.getLogger(__name__)

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0
FINGER_TIP_INDICES = (
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)
NON_THUMB_TIP_INDICES = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_FINGER_JOINTS = {
    "index": (5, 6, INDEX_TIP),
    "middle": (9, 10, MIDDLE_TIP),
    "ring": (13, 14, RING_TIP),
    "pinky": (17, 18, PINKY_TIP),
}


class Gesture(str, Enum):
    """Recognized application gesture."""

    NO_HAND = "No hand"
    POINTING = "Index finger / move"
    PINCH = "Pinch (no click)"
    FIST = "Fist / right click"
    PEACE = "Peace / double click"
    SCROLL_UP = "Thumb up / scroll up"
    SCROLL_DOWN = "Thumb down / scroll down"
    UNKNOWN = "Unknown gesture"

    @property
    def is_click(self) -> bool:
        """Whether this gesture maps to a mouse click action."""
        return self in (Gesture.FIST, Gesture.PEACE)


class ScrollDirection(str, Enum):
    """Direction returned by the optional thumb scroll gesture."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class GestureResult:
    """Classification and supporting geometry for one hand."""

    gesture: Gesture
    finger_states: Mapping[str, bool]
    tip_wrist_distances: Mapping[int, float]
    pinch_distance: float

    @property
    def label(self) -> str:
        """Human-readable gesture label for the OSD."""
        return self.gesture.value


class TapState(str, Enum):
    """State machine states for index-finger tap recognition."""

    EXTENDED = "EXTENDED"
    CURLING = "CURLING"
    TAPPED = "TAPPED"


class TapDetector:
    """Detect a quick downward index-finger curl and require re-arming.

    The detector stores the last ten index-tip Y positions.  A tap is emitted
    only after the finger has been stably extended, moves below its PIP joint
    with sufficient downward velocity, and completes the curl within the
    configured frame window.  Once emitted, the detector must observe the
    finger extended again before another tap can be emitted.
    """

    def __init__(self, settings: TapSettings | None = None) -> None:
        """Initialize the tap state machine.

        Args:
            settings: Tap sensitivity and timing configuration.
        """
        self.settings = settings or TapSettings()
        self.settings.validate()
        self._history: deque[tuple[int, float]] = deque(
            maxlen=self.settings.history_size
        )
        self.state = TapState.EXTENDED
        self._armed = False
        self._extended_frames = 0
        self._curl_start_frame: int | None = None
        self._curl_reference_y: float | None = None
        self._last_extended_y: float | None = None
        self._last_extended_frame: int | None = None
        self._last_frame = -1
        self.last_frame_velocity = 0.0
        self.last_tap_velocity = 0.0
        self.last_displacement = 0.0
        self.last_tap_frame: int | None = None

    @property
    def y_history(self) -> tuple[float, ...]:
        """Return the retained index-tip Y positions, oldest first."""
        return tuple(value for _, value in self._history)

    @property
    def frame_history(self) -> tuple[int, ...]:
        """Return frame numbers corresponding to :attr:`y_history`."""
        return tuple(frame for frame, _ in self._history)

    @property
    def is_armed(self) -> bool:
        """Whether an extended finger can start a new tap."""
        return self._armed

    def _set_state(self, state: TapState) -> None:
        """Set state and log meaningful transitions."""
        if state == self.state:
            return
        LOGGER.debug("Tap detector state: %s -> %s", self.state.value, state.value)
        self.state = state

    def reset(self) -> None:
        """Reset history and require a newly observed extended position."""
        self._history.clear()
        self.state = TapState.EXTENDED
        self._armed = False
        self._extended_frames = 0
        self._curl_start_frame = None
        self._curl_reference_y = None
        self._last_extended_y = None
        self._last_extended_frame = None
        self._last_frame = -1
        self.last_frame_velocity = 0.0
        self.last_tap_velocity = 0.0
        self.last_displacement = 0.0
        self.last_tap_frame = None

    def update_settings(self, settings: TapSettings) -> None:
        """Apply calibration changes without discarding the current history."""
        settings.validate()
        self.settings = settings
        self._history = deque(self._history, maxlen=settings.history_size)

    def _calculate_frame_velocity(
        self, current_frame: int, current_y: float
    ) -> float:
        """Return downward-positive velocity from the previous sample."""
        if len(self._history) < 2:
            return 0.0
        previous_frame, previous_y = self._history[-1]
        elapsed = max(1, current_frame - previous_frame)
        return (current_y - previous_y) / elapsed

    def _begin_curl(
        self, current_frame: int, tip_y: float, reference_y: float | None = None
    ) -> bool:
        """Enter CURLING when the detector is armed."""
        if not self._armed:
            return False
        self._set_state(TapState.CURLING)
        self._curl_start_frame = current_frame
        self._curl_reference_y = (
            reference_y
            if reference_y is not None
            else (self._history[-2][1] if len(self._history) >= 2 else tip_y)
        )
        self.last_displacement = 0.0
        return True

    def _cancel_curl(self) -> None:
        """Cancel a curl and require a fresh stable extended pose."""
        self._armed = False
        self._extended_frames = 0
        self._set_state(TapState.EXTENDED)
        self._curl_start_frame = None
        self._curl_reference_y = None

    def detect_tap_gesture(
        self,
        index_tip_y: float,
        index_pip_y: float,
        frame_number: int | None = None,
        suppress: bool = False,
    ) -> bool:
        """Compatibility-named wrapper around :meth:`update`."""
        return self.update(
            index_tip_y,
            index_pip_y,
            frame_number=frame_number,
            suppress=suppress,
        )

    def update(
        self,
        index_tip_y: float,
        index_pip_y: float,
        frame_number: int | None = None,
        suppress: bool = False,
    ) -> bool:
        """Update the state machine and return ``True`` for one tap edge.

        Args:
            index_tip_y: Normalized image Y coordinate of landmark 8.
            index_pip_y: Normalized image Y coordinate of landmark 6.
            frame_number: Monotonic frame number. When omitted, an internal
                counter is used.
            suppress: Reset/disable detection for conflicting gestures.

        Returns:
            ``True`` only on the frame where a new tap is registered.
        """
        if not self.settings.enabled or suppress:
            self.reset()
            return False
        if not (
            math.isfinite(float(index_tip_y))
            and math.isfinite(float(index_pip_y))
        ):
            self.reset()
            return False
        if frame_number is None:
            current_frame = self._last_frame + 1
        else:
            current_frame = max(int(frame_number), self._last_frame + 1)
        self._last_frame = current_frame
        tip_y = float(index_tip_y)
        pip_y = float(index_pip_y)
        is_extended = tip_y < pip_y - self.settings.extended_y_margin

        self.last_frame_velocity = self._calculate_frame_velocity(
            current_frame, tip_y
        )
        self._history.append((current_frame, tip_y))

        significant_downward = bool(
            is_extended
            and self._last_extended_y is not None
            and self._last_extended_frame is not None
            and self.last_frame_velocity >= self.settings.velocity_threshold
            and tip_y - self._last_extended_y >= self.settings.min_tip_drop
            and current_frame - self._last_extended_frame
            <= self.settings.max_frames_for_tap
        )

        if self.state is TapState.TAPPED:
            if is_extended and abs(self.last_frame_velocity) <= self.settings.velocity_threshold:
                self._extended_frames += 1
            else:
                self._extended_frames = 0
            if is_extended and self._extended_frames >= self.settings.rearm_frames:
                self._armed = True
                self._last_extended_y = tip_y
                self._last_extended_frame = current_frame
                self._set_state(TapState.EXTENDED)
            return False

        if self.state is TapState.EXTENDED:
            if is_extended and not significant_downward:
                if abs(self.last_frame_velocity) <= self.settings.velocity_threshold:
                    self._extended_frames += 1
                    self._last_extended_y = tip_y
                    self._last_extended_frame = current_frame
                else:
                    self._extended_frames = 0
                if self._extended_frames >= self.settings.extended_stable_frames:
                    self._armed = True
                return False
            # Below the PIP, or a significant downward move while still above
            # it, is the beginning of a candidate curl.
            self._begin_curl(current_frame, tip_y, self._last_extended_y)
            return False

        if self.state is TapState.CURLING and is_extended:
            if not significant_downward and self.last_displacement < self.settings.min_tip_drop:
                self._cancel_curl()
                return False

        if self.state is not TapState.CURLING or self._curl_start_frame is None:
            return False
        elapsed = current_frame - self._curl_start_frame
        if self._curl_reference_y is None:
            self._curl_reference_y = tip_y
        self.last_displacement = tip_y - self._curl_reference_y
        self.last_tap_velocity = self.last_displacement / max(1, elapsed)
        curl_sample_count = elapsed + 1
        if (
            self.settings.min_frames_for_tap
            <= curl_sample_count
            <= self.settings.max_frames_for_tap
            and self.last_tap_velocity >= self.settings.velocity_threshold
            and self.last_displacement >= self.settings.min_tip_drop
        ):
            self._armed = False
            self._extended_frames = 0
            self._last_extended_y = None
            self._last_extended_frame = None
            self.last_tap_frame = current_frame
            self._set_state(TapState.TAPPED)
            LOGGER.info(
                "Tap detected at frame %d (velocity=%.4f, drop=%.4f)",
                current_frame,
                self.last_tap_velocity,
                self.last_displacement,
            )
            return True

        if curl_sample_count > self.settings.max_frames_for_tap:
            # A slow curl is deliberately not a tap.  Keep the detector
            # disarmed until the finger visibly returns to EXTENDED.
            self._cancel_curl()
        return False


class PinchDragDetector:
    """Start drag after pinch held, release after pinch lost."""

    def __init__(self, settings: GestureSettings | None = None) -> None:
        """Initialize drag counters."""
        self.settings = settings or GestureSettings()
        self.dragging = False
        self._hold_frames = 0
        self._release_frames = 0

    def update_settings(self, settings: GestureSettings) -> None:
        """Apply new thresholds, keeping drag state."""
        settings.validate()
        self.settings = settings

    def reset(self) -> None:
        """Clear drag state after hand loss."""
        self.dragging = False
        self._hold_frames = 0
        self._release_frames = 0

    def update(self, pinch_active: bool) -> tuple[bool, bool, bool]:
        """Update with pinch state.

        Returns:
            (dragging, started, ended) for this frame.
        """
        if not self.settings.drag_enabled:
            was = self.dragging
            self.reset()
            return False, False, was
        if pinch_active:
            self._release_frames = 0
            self._hold_frames += 1
            if not self.dragging and self._hold_frames >= self.settings.drag_hold_frames:
                self.dragging = True
                return True, True, False
            return self.dragging, False, False
        self._hold_frames = 0
        if not self.dragging:
            self._release_frames = 0
            return False, False, False
        self._release_frames += 1
        if self._release_frames >= self.settings.drag_release_frames:
            self.dragging = False
            self._release_frames = 0
            return False, False, True
        return True, False, False


def _coordinates(landmark: NormalizedLandmark) -> tuple[float, float] | None:
    """Return finite x/y coordinates, or ``None`` for an invalid landmark."""
    if landmark.x is None or landmark.y is None:
        return None
    x = float(landmark.x)
    y = float(landmark.y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def landmark_distance(
    first: NormalizedLandmark, second: NormalizedLandmark
) -> float:
    """Return the normalized Euclidean distance between two landmarks."""
    first_point = _coordinates(first)
    second_point = _coordinates(second)
    if first_point is None or second_point is None:
        return math.inf
    return math.hypot(
        first_point[0] - second_point[0],
        first_point[1] - second_point[1],
    )


def fingertip_wrist_distances(
    landmarks: Sequence[NormalizedLandmark],
) -> dict[int, float]:
    """Calculate the distance from every fingertip to the wrist.

    The returned mapping includes the thumb and all four fingers.  Invalid or
    incomplete landmarks are represented by ``math.inf`` so they cannot satisfy
    a closed-fist threshold.
    """
    distances: dict[int, float] = {}
    if len(landmarks) <= WRIST:
        return {index: math.inf for index in FINGER_TIP_INDICES}
    wrist = landmarks[WRIST]
    for tip_index in FINGER_TIP_INDICES:
        distances[tip_index] = (
            landmark_distance(landmarks[tip_index], wrist)
            if len(landmarks) > tip_index
            else math.inf
        )
    return distances


def _angle_degrees(
    first: NormalizedLandmark, joint: NormalizedLandmark, last: NormalizedLandmark
) -> float:
    """Return the angle at ``joint`` in degrees."""
    first_point = _coordinates(first)
    joint_point = _coordinates(joint)
    last_point = _coordinates(last)
    if first_point is None or joint_point is None or last_point is None:
        return 0.0
    first_vector = (first_point[0] - joint_point[0], first_point[1] - joint_point[1])
    last_vector = (last_point[0] - joint_point[0], last_point[1] - joint_point[1])
    first_norm = math.hypot(*first_vector)
    last_norm = math.hypot(*last_vector)
    if first_norm <= 1e-9 or last_norm <= 1e-9:
        return 0.0
    cosine = (
        first_vector[0] * last_vector[0] + first_vector[1] * last_vector[1]
    ) / (first_norm * last_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def finger_is_extended(
    landmarks: Sequence[NormalizedLandmark],
    finger_name: str,
    settings: GestureSettings,
) -> bool:
    """Estimate whether one non-thumb finger is extended.

    A straightness test and a distance-from-wrist test are combined.  The
    distance test remains useful when a finger is foreshortened by the camera,
    while the angle test prevents curled fingertips from being called open.
    """
    try:
        mcp_index, pip_index, tip_index = _FINGER_JOINTS[finger_name]
    except KeyError as exc:
        raise ValueError(f"Unknown finger name: {finger_name}") from exc
    if len(landmarks) <= max(mcp_index, pip_index, tip_index):
        return False
    mcp = landmarks[mcp_index]
    pip = landmarks[pip_index]
    tip = landmarks[tip_index]
    wrist = landmarks[WRIST] if len(landmarks) > WRIST else mcp
    tip_distance = landmark_distance(tip, wrist)
    pip_distance = landmark_distance(pip, wrist)
    angle = _angle_degrees(mcp, pip, tip)
    # Prefer a genuinely straight joint chain.  The larger distance-only
    # fallback handles fingers that are strongly foreshortened toward the
    # camera, while avoiding the common false positive where a curled tip is
    # slightly farther from the wrist than its PIP joint.
    strong_distance_extension = tip_distance > pip_distance * (
        settings.finger_extension_ratio + 0.25
    )
    return bool(
        (
            angle >= settings.finger_angle_threshold_degrees
            and tip_distance > pip_distance
        )
        or strong_distance_extension
    )


def finger_states(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> dict[str, bool]:
    """Return extension state for index, middle, ring, and pinky fingers."""
    return {
        finger_name: finger_is_extended(landmarks, finger_name, settings)
        for finger_name in _FINGER_JOINTS
    }


def is_fist(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> bool:
    """Return whether every fingertip is close enough to the wrist.

    This intentionally follows the requested all-fingers-closed rule: the
    thumb and all four fingertips must each be below
    ``fist_tip_wrist_threshold``.
    """
    distances = fingertip_wrist_distances(landmarks)
    if not distances:
        return False
    return all(
        distance < settings.fist_tip_wrist_threshold
        for distance in distances.values()
    )


def is_pinch(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> bool:
    """Return whether the thumb and index fingertips form a pinch."""
    if len(landmarks) <= max(THUMB_TIP, INDEX_TIP):
        return False
    return (
        landmark_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        <= settings.pinch_threshold
    )


def is_peace(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> bool:
    """Return whether index and middle form a two-finger peace sign."""
    if len(landmarks) <= max(NON_THUMB_TIP_INDICES):
        return False
    states = finger_states(landmarks, settings)
    if not (states["index"] and states["middle"]):
        return False
    if states["ring"] or states["pinky"]:
        return False
    separation = landmark_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
    return separation >= settings.peace_min_separation


def is_pointing(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> bool:
    """Return whether the index is extended while the other fingers are curled."""
    if len(landmarks) <= max(NON_THUMB_TIP_INDICES):
        return False
    states = finger_states(landmarks, settings)
    return states["index"] and not any(
        states[name] for name in ("middle", "ring", "pinky")
    )


def scroll_direction(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> ScrollDirection | None:
    """Return a thumb-up or thumb-down scroll direction when applicable."""
    if not settings.scroll_enabled or len(landmarks) <= max(THUMB_TIP, WRIST):
        return None
    states = finger_states(landmarks, settings)
    if any(states[name] for name in ("index", "middle", "ring", "pinky")):
        return None
    thumb_tip = landmarks[THUMB_TIP]
    wrist = landmarks[WRIST]
    thumb_point = _coordinates(thumb_tip)
    wrist_point = _coordinates(wrist)
    if thumb_point is None or wrist_point is None:
        return None
    vertical_delta = thumb_point[1] - wrist_point[1]
    if vertical_delta < -settings.scroll_deadzone:
        return ScrollDirection.UP
    if vertical_delta > settings.scroll_deadzone:
        return ScrollDirection.DOWN
    return None


def classify_gesture(
    landmarks: Sequence[NormalizedLandmark], settings: GestureSettings
) -> GestureResult:
    """Classify one hand using the configured gesture thresholds.

    Gesture priority is fist, pinch, peace, scroll, then pointing.  Fist has
    priority over pinch so a closed hand remains a right-click gesture even
    when the thumb happens to touch the index finger.
    """
    if len(landmarks) < 21:
        return GestureResult(
            gesture=Gesture.UNKNOWN,
            finger_states={},
            tip_wrist_distances={},
            pinch_distance=math.inf,
        )

    states = finger_states(landmarks, settings)
    distances = fingertip_wrist_distances(landmarks)
    pinch_distance = landmark_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])

    if is_fist(landmarks, settings):
        gesture = Gesture.FIST
    elif states["index"] and is_pinch(landmarks, settings):
        gesture = Gesture.PINCH
    elif is_peace(landmarks, settings):
        gesture = Gesture.PEACE
    else:
        direction = scroll_direction(landmarks, settings)
        if direction is ScrollDirection.UP:
            gesture = Gesture.SCROLL_UP
        elif direction is ScrollDirection.DOWN:
            gesture = Gesture.SCROLL_DOWN
        elif is_pointing(landmarks, settings):
            gesture = Gesture.POINTING
        else:
            gesture = Gesture.UNKNOWN

    return GestureResult(
        gesture=gesture,
        finger_states=states,
        tip_wrist_distances=distances,
        pinch_distance=pinch_distance,
    )


class GestureRecognizer:
    """Stateless gesture classifier with a stateful tap detector."""

    def __init__(
        self,
        settings: GestureSettings | None = None,
        tap_settings: TapSettings | None = None,
    ) -> None:
        """Initialize gesture and tap settings."""
        self.settings = settings or GestureSettings()
        self.settings.validate()
        self.tap_detector = TapDetector(tap_settings)
        self.drag_detector = PinchDragDetector(self.settings)
        self._last_scroll: ScrollDirection | None = None

    def recognize(
        self, landmarks: Sequence[NormalizedLandmark]
    ) -> GestureResult:
        """Classify a sequence of MediaPipe landmarks."""
        return classify_gesture(landmarks, self.settings)

    def detect_tap_gesture(
        self,
        landmarks: Sequence[NormalizedLandmark],
        frame_number: int | None = None,
        suppress: bool = False,
    ) -> bool:
        """Update and query the index-finger tap detector.

        Args:
            landmarks: A complete MediaPipe hand landmark sequence.
            frame_number: Monotonic frame number for velocity calculations.
            suppress: Suppress/reset detection for a conflicting gesture.

        Returns:
            ``True`` only on a newly registered tap edge.
        """
        if len(landmarks) <= max(INDEX_TIP, 6):
            self.tap_detector.reset()
            return False
        index_tip = landmarks[INDEX_TIP]
        index_pip = landmarks[6]
        if index_tip.y is None or index_pip.y is None:
            self.tap_detector.reset()
            return False
        return self.tap_detector.update(
            index_tip.y,
            index_pip.y,
            frame_number=frame_number,
            suppress=suppress,
        )

    def reset_tap(self) -> None:
        """Clear tap history after hand loss or a tracking reset."""
        self.tap_detector.reset()
        self.drag_detector.reset()
        self._last_scroll = None

    def detect_drag(
        self, landmarks: Sequence[NormalizedLandmark]
    ) -> tuple[bool, bool, bool]:
        """Update pinch-hold drag state, return (dragging, started, ended)."""
        if len(landmarks) < 21:
            self.drag_detector.reset()
            return False, False, False
        pinch = is_pinch(landmarks, self.settings)
        # Only index-extended pinch counts as drag (matches click_action_for).
        states = finger_states(landmarks, self.settings)
        if not states.get("index", False):
            pinch = False
        return self.drag_detector.update(pinch)

    def scroll_with_hysteresis(
        self, landmarks: Sequence[NormalizedLandmark]
    ) -> ScrollDirection | None:
        """Return scroll direction with hysteresis to avoid flicker."""
        raw = scroll_direction(landmarks, self.settings)
        if raw is not None:
            self._last_scroll = raw
            return raw
        if self._last_scroll is None or len(landmarks) <= max(THUMB_TIP, WRIST):
            return None
        # Stay in scroll while thumb remains beyond deadzone minus hysteresis.
        thumb_tip = landmarks[THUMB_TIP]
        wrist = landmarks[WRIST]
        thumb_point = _coordinates(thumb_tip)
        wrist_point = _coordinates(wrist)
        if thumb_point is None or wrist_point is None:
            self._last_scroll = None
            return None
        vertical_delta = thumb_point[1] - wrist_point[1]
        hold_band = max(0.0, self.settings.scroll_deadzone - self.settings.scroll_hysteresis)
        if self._last_scroll is ScrollDirection.UP and vertical_delta < -hold_band:
            return ScrollDirection.UP
        if self._last_scroll is ScrollDirection.DOWN and vertical_delta > hold_band:
            return ScrollDirection.DOWN
        self._last_scroll = None
        return None

    def update_settings(
        self,
        settings: GestureSettings,
        tap_settings: TapSettings | None = None,
    ) -> None:
        """Replace settings after a configuration reload or adjustment."""
        settings.validate()
        self.settings = settings
        self.drag_detector.update_settings(settings)
        if tap_settings is not None:
            self.tap_detector.update_settings(tap_settings)
