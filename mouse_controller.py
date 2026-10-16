"""Mouse input, screen-region mapping, smoothing, and click debouncing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from config import MouseSettings

LOGGER = logging.getLogger(__name__)


class ClickAction(str, Enum):
    """Mouse click actions emitted by gesture classification."""

    LEFT_CLICK = "left click"
    RIGHT_CLICK = "right click"
    DOUBLE_CLICK = "double click"


class ClickDebouncer:
    """Turn stable gesture edges into rate-limited click events.

    A gesture must persist for ``stable_frames`` and must be released before a
    second event is emitted.  The global cooldown prevents rapid alternating
    gestures from producing click bursts while still allowing different click
    types to be used in sequence.
    """

    def __init__(self, cooldown_frames: int = 20, stable_frames: int = 2) -> None:
        """Initialize a frame-based debouncer."""
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames cannot be negative")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self.cooldown_frames = cooldown_frames
        self.stable_frames = stable_frames
        self._candidate: ClickAction | None = None
        self._candidate_frames = 0
        self._active_action: ClickAction | None = None
        self._last_click_frame: int | None = None

    @property
    def last_click_frame(self) -> int | None:
        """Frame number of the last emitted click, if any."""
        return self._last_click_frame

    def frames_until_ready(self, frame_number: int) -> int:
        """Return the number of cooldown frames remaining."""
        if self._last_click_frame is None:
            return 0
        return max(0, self.cooldown_frames - (frame_number - self._last_click_frame))

    def update(
        self, action: ClickAction | None, frame_number: int
    ) -> ClickAction | None:
        """Process one frame and return a click only on a valid gesture edge.

        Args:
            action: Candidate action for this frame, or ``None`` when no click
                gesture is active.
            frame_number: Monotonically increasing frame counter.

        Returns:
            The action to execute, or ``None`` when no click should occur.
        """
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")

        if action is None:
            self._candidate = None
            self._candidate_frames = 0
            self._active_action = None
            return None

        if action != self._candidate:
            self._candidate = action
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        # Once an action is armed, holding the same gesture cannot emit another
        # click.  It must be released (or replaced by a different action) first.
        if self._active_action == action:
            return None
        if self._candidate_frames < self.stable_frames:
            return None

        # Arm the action even when cooldown blocks it.  This prevents a held
        # gesture from unexpectedly firing once the cooldown expires.
        self._active_action = action
        if self.frames_until_ready(frame_number) > 0:
            return None
        self._last_click_frame = frame_number
        return action

    def emit_edge(
        self, action: ClickAction, frame_number: int
    ) -> ClickAction | None:
        """Emit a one-frame click edge while respecting global cooldown.

        Tap detection is already an edge event, so it must not wait for the
        multi-frame stable-pose confirmation used by held gestures.
        """
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        if self.frames_until_ready(frame_number) > 0:
            return None
        self._last_click_frame = frame_number
        return action

    def reset(self) -> None:
        """Clear gesture state while retaining cooldown timing."""
        self._candidate = None
        self._candidate_frames = 0
        self._active_action = None


@dataclass(frozen=True)
class ScreenPoint:
    """A point in desktop pixel coordinates."""

    x: int
    y: int


class OneEuroFilter:
    """Adaptive low-pass filter for jitter-free cursor motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, dcutoff: float = 1.0) -> None:
        """Initialize filter state."""
        if min_cutoff <= 0.0 or dcutoff <= 0.0 or beta < 0.0:
            raise ValueError("OneEuro parameters must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: float | None = None
        self._dx: float = 0.0
        self._last_time: float | None = None

    def _alpha(self, cutoff: float, rate: float) -> float:
        """Compute smoothing factor for a cutoff frequency."""
        import math

        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * rate)

    def filter(self, value: float, timestamp: float | None = None) -> float:
        """Filter one sample."""
        import math
        import time

        now = timestamp if timestamp is not None else time.perf_counter()
        if self._x is None or self._last_time is None:
            self._x = value
            self._last_time = now
            self._dx = 0.0
            return value
        rate = 1.0 / max(1e-6, now - self._last_time)
        self._last_time = now
        d_cutoff_alpha = self._alpha(self.dcutoff, rate)
        dx = (value - self._x) * rate
        self._dx = d_cutoff_alpha * dx + (1.0 - d_cutoff_alpha) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        alpha = self._alpha(cutoff, rate)
        self._x = alpha * value + (1.0 - alpha) * self._x
        return self._x

    def reset(self) -> None:
        """Clear filter state."""
        self._x = None
        self._dx = 0.0
        self._last_time = None


class DwellClickDetector:
    """Emit a left click after hovering stationary."""

    def __init__(self, dwell_time_ms: int = 600, dwell_radius_px: int = 12) -> None:
        """Initialize dwell timing."""
        self.dwell_time_ms = dwell_time_ms
        self.dwell_radius_px = dwell_radius_px
        self._anchor: ScreenPoint | None = None
        self._anchor_time: float | None = None
        self._fired = False

    def update(self, point: ScreenPoint, now: float | None = None) -> bool:
        """Update with a cursor point, return True on dwell edge."""
        import math
        import time

        current = now if now is not None else time.perf_counter()
        if self._anchor is None or self._anchor_time is None:
            self._anchor = point
            self._anchor_time = current
            self._fired = False
            return False
        dist = math.hypot(point.x - self._anchor.x, point.y - self._anchor.y)
        if dist > self.dwell_radius_px:
            self._anchor = point
            self._anchor_time = current
            self._fired = False
            return False
        elapsed_ms = (current - self._anchor_time) * 1000.0
        if not self._fired and elapsed_ms >= self.dwell_time_ms:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        """Clear dwell state."""
        self._anchor = None
        self._anchor_time = None
        self._fired = False


class MouseController:
    """Translate normalized index-finger coordinates into mouse actions."""

    def __init__(
        self,
        settings: MouseSettings,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the controller.

        Args:
            settings: Cursor mapping and smoothing settings.
            screen_size: Optional ``(width, height)`` override.  When omitted,
                PyAutoGUI is queried for the primary screen size.
        """
        settings.validate()
        self.settings = settings
        self._screen_size = screen_size or self._query_screen_size()
        if self._screen_size[0] <= 0 or self._screen_size[1] <= 0:
            raise RuntimeError("Screen dimensions must be positive")
        self._smoothed_position: tuple[float, float] | None = None
        self._last_position: ScreenPoint | None = None
        self._pyautogui: ModuleType | Any | None = None
        self._movement_error_logged = False
        self._one_euro_x = OneEuroFilter(
            settings.one_euro_min_cutoff, settings.one_euro_beta, settings.one_euro_dcutoff
        )
        self._one_euro_y = OneEuroFilter(
            settings.one_euro_min_cutoff, settings.one_euro_beta, settings.one_euro_dcutoff
        )
        self._dragging = False

    @staticmethod
    def _query_screen_size() -> tuple[int, int]:
        """Query PyAutoGUI while producing a useful error if unavailable."""
        try:
            import pyautogui

            # PyAutoGUI's historical default PAUSE=0.1 seconds adds latency to
            # every movement/click command.  The application provides its own
            # frame/cooldown throttling, so keep the low-level calls immediate.
            pyautogui.PAUSE = 0
            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception as exc:  # PyAutoGUI has platform-specific exceptions.
            raise RuntimeError(
                "Unable to query the desktop screen through PyAutoGUI. "
                "A desktop session is required."
            ) from exc

    @property
    def screen_size(self) -> tuple[int, int]:
        """Return the detected screen size."""
        return self._screen_size

    @property
    def position(self) -> ScreenPoint | None:
        """Return the last emitted screen point, if movement has occurred."""
        return self._last_position

    def update_settings(self, settings: MouseSettings) -> None:
        """Apply runtime-updated mouse settings without resetting position."""
        settings.validate()
        self.settings = settings
        self._one_euro_x = OneEuroFilter(
            settings.one_euro_min_cutoff, settings.one_euro_beta, settings.one_euro_dcutoff
        )
        self._one_euro_y = OneEuroFilter(
            settings.one_euro_min_cutoff, settings.one_euro_beta, settings.one_euro_dcutoff
        )

    def set_smoothing_factor(self, factor: float) -> None:
        """Set the exponential smoothing factor at runtime."""
        if factor < 1.0:
            raise ValueError("smoothing factor must be at least 1.0")
        self.settings.smoothing_factor = factor

    def map_to_screen(
        self, normalized_x: float, normalized_y: float
    ) -> tuple[float, float]:
        """Map normalized camera coordinates into the configured screen region."""
        if not (
            math.isfinite(normalized_x)
            and math.isfinite(normalized_y)
        ):
            raise ValueError("normalized coordinates must be finite")
        x = max(0.0, min(1.0, normalized_x))
        y = max(0.0, min(1.0, normalized_y))
        region = self.settings.screen_region
        screen_width, screen_height = self._screen_size
        left = screen_width * region.x
        top = screen_height * region.y
        mapped_x = left + x * screen_width * region.width
        mapped_y = top + y * screen_height * region.height

        # Movement scale expands around the center of the selected region,
        # which preserves a natural zero point and then clamps to the screen.
        center_x = left + screen_width * region.width / 2.0
        center_y = top + screen_height * region.height / 2.0
        mapped_x = center_x + (mapped_x - center_x) * self.settings.movement_scale
        mapped_y = center_y + (mapped_y - center_y) * self.settings.movement_scale
        return (
            max(0.0, min(float(screen_width - 1), mapped_x)),
            max(0.0, min(float(screen_height - 1), mapped_y)),
        )

    def move_to_normalized(
        self, normalized_x: float, normalized_y: float
    ) -> ScreenPoint:
        """Smooth and emit one cursor movement command.

        Supports ``exponential`` (``current += (target-current)/factor``) and
        ``one_euro`` adaptive filtering. A small pixel dead zone suppresses
        residual model jitter.
        """
        import time

        target_x, target_y = self.map_to_screen(normalized_x, normalized_y)
        if self.settings.smoothing_backend == "one_euro":
            now = time.perf_counter()
            smoothed_x = float(self._one_euro_x.filter(target_x, now))
            smoothed_y = float(self._one_euro_y.filter(target_y, now))
            if self._smoothed_position is not None:
                current_x, current_y = self._smoothed_position
                if math.hypot(smoothed_x - current_x, smoothed_y - current_y) < self.settings.dead_zone_pixels:
                    point = self._last_position or ScreenPoint(
                        int(round(current_x)), int(round(current_y))
                    )
                    self._last_position = point
                    return point
        elif self._smoothed_position is None:
            smoothed_x, smoothed_y = target_x, target_y
        else:
            current_x, current_y = self._smoothed_position
            delta_x = target_x - current_x
            delta_y = target_y - current_y
            if math.hypot(delta_x, delta_y) < self.settings.dead_zone_pixels:
                point = self._last_position or ScreenPoint(
                    int(round(current_x)), int(round(current_y))
                )
                self._last_position = point
                return point
            factor = max(1.0, self.settings.smoothing_factor)
            smoothed_x = current_x + delta_x / factor
            smoothed_y = current_y + delta_y / factor

        self._smoothed_position = (smoothed_x, smoothed_y)
        point = ScreenPoint(
            int(round(max(0.0, min(self._screen_size[0] - 1, smoothed_x)))),
            int(round(max(0.0, min(self._screen_size[1] - 1, smoothed_y)))),
        )
        self._emit_movement(point)
        self._last_position = point
        return point

    def _load_pyautogui(self) -> Any:
        """Import and cache PyAutoGUI lazily."""
        if self._pyautogui is None:
            try:
                import pyautogui
            except ImportError as exc:
                raise RuntimeError(
                    "PyAutoGUI is required for mouse control; install requirements.txt"
                ) from exc
            pyautogui.PAUSE = 0
            self._pyautogui = pyautogui
        return self._pyautogui

    def _emit_movement(self, point: ScreenPoint) -> None:
        """Send a movement command, logging repeated desktop failures once."""
        try:
            self._load_pyautogui().moveTo(point.x, point.y)
            self._movement_error_logged = False
        except Exception as exc:
            if not self._movement_error_logged:
                LOGGER.error("Unable to move the mouse cursor: %s", exc)
                self._movement_error_logged = True

    def perform_click(self, action: ClickAction) -> bool:
        """Execute a click action through PyAutoGUI."""
        try:
            pyautogui = self._load_pyautogui()
            if action is ClickAction.LEFT_CLICK:
                pyautogui.click()
            elif action is ClickAction.RIGHT_CLICK:
                pyautogui.rightClick()
            elif action is ClickAction.DOUBLE_CLICK:
                pyautogui.doubleClick()
            else:
                raise ValueError(f"Unsupported click action: {action}")
            LOGGER.debug("Executed %s", action.value)
            return True
        except Exception as exc:
            LOGGER.error("Unable to execute %s: %s", action.value, exc)
            return False

    @property
    def is_dragging(self) -> bool:
        """Whether a drag (mouse button held) is active."""
        return self._dragging

    def set_dragging(self, dragging: bool) -> bool:
        """Press or release the left button for drag gestures."""
        if dragging == self._dragging:
            return True
        try:
            pyautogui = self._load_pyautogui()
            if dragging:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()
            self._dragging = dragging
            LOGGER.info("Drag %s", "started" if dragging else "ended")
            return True
        except Exception as exc:
            LOGGER.error("Unable to set drag=%s: %s", dragging, exc)
            return False

    def scroll(self, direction: str, amount: int) -> bool:
        """Scroll up or down by a positive integer amount."""
        if amount <= 0:
            raise ValueError("scroll amount must be positive")
        try:
            pyautogui = self._load_pyautogui()
            scroll_value = abs(amount) if direction.lower() == "up" else -abs(amount)
            pyautogui.scroll(scroll_value)
            return True
        except Exception as exc:
            LOGGER.error("Unable to scroll mouse: %s", exc)
            return False
