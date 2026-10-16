"""Pluggable mouse backends: PyAutoGUI (X11), xdotool (X11 low-latency), fake (tests).

God-level requires <30ms input latency and Wayland/X11 portability.
PyAutoGUI goes through Xlib/XTEST IPC (~5-15ms). xdotool calls xdo directly
and is measurably faster on X11. Wayland compositors need ydotool/wtype;
this module auto-detects and falls back safely so the app never crashes
for lack of a backend.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Monitor:
    """One physical monitor in desktop pixels."""

    x: int
    y: int
    width: int
    height: int
    primary: bool = False


def list_monitors() -> list[Monitor]:
    """Enumerate monitors via xrandr, fallback to single primary."""
    try:
        out = subprocess.run(
            ["xrandr", "--listmonitors"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            monitors: list[Monitor] = []
            for line in out.stdout.splitlines()[1:]:
                # e.g: " 0: +*selkies-1920x850 1920/406x850/180+0+0  ..."
                parts = line.split()
                if len(parts) < 3:
                    continue
                name_geo = parts[2] if parts[1].startswith("+") else parts[1]
                primary = "*" in parts[1]
                # geo like 1920/406x850/180+0+0 -> 1920x850+0+0
                geo = name_geo.split("/")[-1] if "/" in name_geo else name_geo
                # split WxH+X+Y
                try:
                    wh, *xy = geo.replace("x", " ").replace("+", " ").split()
                    # actually "1920x850+0+0" -> ["1920","850","0","0"]
                    nums = geo.replace("x", " ").replace("+", " ").split()
                    w, h, x, y = (int(float(n)) for n in nums[:4])
                    monitors.append(Monitor(x, y, w, h, primary))
                except (ValueError, IndexError):
                    continue
            if monitors:
                return monitors
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    # Fallback: single primary from PyAutoGUI or 1920x1080 default
    try:
        import pyautogui

        size = pyautogui.size()
        return [Monitor(0, 0, int(size.width), int(size.height), True)]
    except Exception:
        return [Monitor(0, 0, 1920, 1080, True)]


def virtual_desktop_size(monitors: list[Monitor] | None = None) -> tuple[int, int]:
    """Return bounding-box size of all monitors."""
    monitors = monitors or list_monitors()
    if not monitors:
        return 1920, 1080
    right = max(m.x + m.width for m in monitors)
    bottom = max(m.y + m.height for m in monitors)
    left = min(m.x for m in monitors)
    top = min(m.y for m in monitors)
    return max(1, right - left), max(1, bottom - top)


class MouseBackend:
    """Backend protocol: move, click, mouseDown/Up, scroll, size."""

    name = "base"

    def size(self) -> tuple[int, int]:
        """Return (width, height) of virtual desktop."""
        raise NotImplementedError

    def move_to(self, x: int, y: int) -> None:
        """Move cursor to desktop pixels."""
        raise NotImplementedError

    def click(self) -> None:
        """Left click."""
        raise NotImplementedError

    def right_click(self) -> None:
        """Right click."""
        raise NotImplementedError

    def double_click(self) -> None:
        """Double click."""
        raise NotImplementedError

    def middle_click(self) -> None:
        """Middle click (second-hand pinch)."""
        raise NotImplementedError

    def mouse_down(self) -> None:
        """Press left button."""
        raise NotImplementedError

    def mouse_up(self) -> None:
        """Release left button."""
        raise NotImplementedError

    def scroll(self, amount: int) -> None:
        """Scroll vertically (positive up)."""
        raise NotImplementedError

    def hotkey(self, *keys: str) -> None:
        """Press key combination."""
        raise NotImplementedError


class PyAutoGUIBackend(MouseBackend):
    """Default backend via PyAutoGUI (X11/XTEST)."""

    name = "pyautogui"

    def __init__(self) -> None:
        """Import PyAutoGUI lazily and disable its pause."""
        import pyautogui

        pyautogui.PAUSE = 0
        self._pg = pyautogui

    def size(self) -> tuple[int, int]:
        """Return virtual desktop size across monitors."""
        monitors = list_monitors()
        # If xrandr found real monitors, use bounding box (multi-monitor).
        if len(monitors) > 1 or (monitors and (monitors[0].width, monitors[0].height) != (1920, 1080)):
            return virtual_desktop_size(monitors)
        size = self._pg.size()
        return int(size.width), int(size.height)

    def move_to(self, x: int, y: int) -> None:
        """Move cursor."""
        self._pg.moveTo(x, y)

    def click(self) -> None:
        """Left click."""
        self._pg.click()

    def right_click(self) -> None:
        """Right click."""
        self._pg.rightClick()

    def double_click(self) -> None:
        """Double click."""
        self._pg.doubleClick()

    def middle_click(self) -> None:
        """Middle click."""
        self._pg.middleClick()

    def mouse_down(self) -> None:
        """Press left button."""
        self._pg.mouseDown()

    def mouse_up(self) -> None:
        """Release left button."""
        self._pg.mouseUp()

    def scroll(self, amount: int) -> None:
        """Scroll vertically."""
        self._pg.scroll(amount)

    def hotkey(self, *keys: str) -> None:
        """Press key combination."""
        self._pg.hotkey(*keys)


class XDotoolBackend(MouseBackend):
    """Low-latency X11 backend via xdotool (requires `xdotool` binary)."""

    name = "xdotool"

    def __init__(self) -> None:
        """Check xdotool availability."""
        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool binary not found")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAY not set for xdotool")

    def _run(self, *args: str) -> None:
        """Run xdotool, raising on failure."""
        subprocess.run(["xdotool", *args], check=True, timeout=5)

    def size(self) -> tuple[int, int]:
        """Return virtual desktop size."""
        return virtual_desktop_size()

    def move_to(self, x: int, y: int) -> None:
        """Move cursor."""
        self._run("mousemove", str(x), str(y))

    def click(self) -> None:
        """Left click."""
        self._run("click", "1")

    def right_click(self) -> None:
        """Right click."""
        self._run("click", "3")

    def double_click(self) -> None:
        """Double click."""
        self._run("click", "--repeat", "2", "1")

    def middle_click(self) -> None:
        """Middle click."""
        self._run("click", "2")

    def mouse_down(self) -> None:
        """Press left button."""
        self._run("mousedown", "1")

    def mouse_up(self) -> None:
        """Release left button."""
        self._run("mouseup", "1")

    def scroll(self, amount: int) -> None:
        """Scroll vertically via buttons 4/5."""
        button = "4" if amount > 0 else "5"
        for _ in range(abs(amount)):
            self._run("click", button)

    def hotkey(self, *keys: str) -> None:
        """Press key combination."""
        self._run("key", "+".join(keys))


class FakeBackend(MouseBackend):
    """In-memory backend for unit tests."""

    name = "fake"

    def __init__(self, width: int = 800, height: int = 600) -> None:
        """Initialize event log."""
        self._size = (width, height)
        self.events: list[tuple[str, tuple]] = []

    def size(self) -> tuple[int, int]:
        """Return fake size."""
        return self._size

    def move_to(self, x: int, y: int) -> None:
        """Record move."""
        self.events.append(("move", (x, y)))

    def click(self) -> None:
        """Record click."""
        self.events.append(("click", ()))

    def right_click(self) -> None:
        """Record right click."""
        self.events.append(("right_click", ()))

    def double_click(self) -> None:
        """Record double click."""
        self.events.append(("double_click", ()))

    def middle_click(self) -> None:
        """Record middle click."""
        self.events.append(("middle_click", ()))

    def mouse_down(self) -> None:
        """Record mouse down."""
        self.events.append(("mouse_down", ()))

    def mouse_up(self) -> None:
        """Record mouse up."""
        self.events.append(("mouse_up", ()))

    def scroll(self, amount: int) -> None:
        """Record scroll."""
        self.events.append(("scroll", (amount,)))

    def hotkey(self, *keys: str) -> None:
        """Record hotkey."""
        self.events.append(("hotkey", keys))


def auto_backend(preferred: str = "auto") -> MouseBackend:
    """Select best available backend.

    Args:
        preferred: ``auto``, ``pyautogui``, ``xdotool``, or ``fake``.
    """
    if preferred == "fake":
        return FakeBackend()
    if preferred == "xdotool":
        return XDotoolBackend()
    if preferred == "pyautogui":
        return PyAutoGUIBackend()
    # auto: xdotool first on X11 (lower latency), fallback to pyautogui
    if os.environ.get("DISPLAY") and shutil.which("xdotool"):
        try:
            backend = XDotoolBackend()
            backend.size()
            LOGGER.info("Mouse backend: xdotool (low-latency X11)")
            return backend
        except Exception as exc:
            LOGGER.warning("xdotool unavailable, falling back to pyautogui: %s", exc)
    LOGGER.info("Mouse backend: pyautogui")
    return PyAutoGUIBackend()
