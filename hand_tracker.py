"""MediaPipe Tasks hand tracking and model management.

This module isolates all interaction with the modern MediaPipe Tasks API.  The
application uses ``VIDEO`` running mode because OpenCV already supplies a
serialized camera loop; this avoids races between an asynchronous callback and
mouse actions while still providing MediaPipe's temporal tracking.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)

LOGGER = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_MIN_SIZE = 1_000_000
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
)


class ModelDownloadError(RuntimeError):
    """Raised when the Hand Landmarker model cannot be downloaded."""


@dataclass(frozen=True)
class HandObservation:
    """A single detected hand and its display metadata."""

    landmarks: tuple[NormalizedLandmark, ...]
    handedness: str = "Unknown"
    confidence: float = 0.0


def _map_landmark_to_full_frame(
    landmark: NormalizedLandmark,
    x_offset: int,
    y_offset: int,
    roi_width: int,
    roi_height: int,
    frame_width: int,
    frame_height: int,
) -> NormalizedLandmark:
    """Map one ROI-normalized landmark back to full-frame coordinates."""
    if landmark.x is None or landmark.y is None:
        return landmark
    return NormalizedLandmark(
        x=(x_offset + landmark.x * roi_width) / frame_width,
        y=(y_offset + landmark.y * roi_height) / frame_height,
        z=landmark.z,
        visibility=landmark.visibility,
        presence=landmark.presence,
        name=landmark.name,
    )


def sha256_of_file(path: str | Path) -> str:
    """Return the hex SHA256 digest of a file."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    model_path: str | Path,
    model_url: str = MODEL_URL,
    timeout_seconds: float = 60.0,
    force: bool = False,
    expected_sha256: str = "",
) -> Path:
    """Download the MediaPipe Hand Landmarker model if necessary.

    The download is written to a temporary file and atomically renamed only
    after the complete response has been received.  This prevents a partial
    model from being used after an interrupted download.

    Args:
        model_path: Destination path for the ``.task`` file.
        model_url: Source URL for the model.
        timeout_seconds: Socket timeout for the HTTP request.
        force: Re-download even when a non-empty model already exists.
        expected_sha256: Optional hex digest verified after download.

    Returns:
        The path to the available model.

    Raises:
        ModelDownloadError: If the request fails or the response is invalid.
    """
    destination = Path(model_path).expanduser()
    if (
        not force
        and destination.exists()
        and destination.stat().st_size >= MODEL_MIN_SIZE
    ):
        LOGGER.debug("Using existing Hand Landmarker model: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        model_url,
        headers={"User-Agent": "Virtual-Mouse/1.0"},
    )
    LOGGER.info("Downloading Hand Landmarker model from %s", model_url)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                expected_size = int(response.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                expected_size = 0
            downloaded = 0
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if expected_size and downloaded % (5 * 1024 * 1024) < len(chunk):
                        LOGGER.info(
                            "Model download progress: %.1f/%.1f MiB",
                            downloaded / (1024 * 1024),
                            expected_size / (1024 * 1024),
                        )
        if downloaded < MODEL_MIN_SIZE:
            raise ModelDownloadError(
                f"Downloaded model is unexpectedly small ({downloaded} bytes)"
            )
        if expected_size and downloaded != expected_size:
            raise ModelDownloadError(
                "Incomplete model download: received "
                f"{downloaded} of {expected_size} bytes"
            )
        os.replace(temporary, destination)
    except ModelDownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Unable to download Hand Landmarker model from {model_url}: {exc}"
        ) from exc
    if expected_sha256:
        actual = sha256_of_file(destination)
        if actual.lower() != expected_sha256.strip().lower():
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Model SHA256 mismatch: expected {expected_sha256[:16]}..., got {actual[:16]}..."
            )
        LOGGER.info("Model SHA256 verified: %s", actual[:16])
    else:
        LOGGER.info(
            "Hand Landmarker model ready: %s (sha256=%s)",
            destination,
            sha256_of_file(destination)[:16],
        )
        return destination
    LOGGER.info("Hand Landmarker model ready: %s", destination)
    return destination


class HandTracker:
    """Create and operate a MediaPipe Tasks ``HandLandmarker``.

    The tracker owns a single task instance and must be closed, preferably by
    using it as a context manager.  ``detect`` accepts OpenCV BGR frames and
    converts them to the RGB ``Image`` container expected by MediaPipe.
    """

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a VIDEO-mode Hand Landmarker task.

        Args:
            model_path: Path to a local ``hand_landmarker.task`` file.
            num_hands: Maximum number of hands to detect.
            min_detection_confidence: Hand detection confidence threshold.
            min_presence_confidence: Hand presence confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
            use_gpu: Try MediaPipe's GPU delegate first and fall back to CPU
                when the installed native build does not support it.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            RuntimeError: If MediaPipe cannot create the task.
        """
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Hand Landmarker model not found: {resolved_path}")
        if resolved_path.stat().st_size < MODEL_MIN_SIZE:
            raise RuntimeError(f"Hand Landmarker model is incomplete: {resolved_path}")

        # VIDEO mode is deliberate here: the OpenCV loop is already serialized,
        # and a LIVE_STREAM callback would introduce stale-result synchronization
        # without improving the single-camera control loop.
        delegates = [BaseOptions.Delegate.CPU]
        if use_gpu:
            delegates.insert(0, BaseOptions.Delegate.GPU)

        landmarker: HandLandmarker | None = None
        last_error: Exception | None = None
        for delegate in delegates:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(resolved_path), delegate=delegate
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            try:
                landmarker = HandLandmarker.create_from_options(options)
                if delegate is BaseOptions.Delegate.GPU:
                    LOGGER.info("MediaPipe GPU delegate enabled")
                break
            except Exception as exc:  # Native builds expose varied errors.
                last_error = exc
                if delegate is BaseOptions.Delegate.GPU:
                    LOGGER.warning(
                        "MediaPipe GPU delegate unavailable; falling back to CPU: %s",
                        exc,
                    )
        if landmarker is None:
            raise RuntimeError(
                f"Unable to initialize MediaPipe HandLandmarker: {last_error}"
            ) from last_error
        self._landmarker = landmarker

        self._last_timestamp_ms = -1
        self._closed = False
        LOGGER.info(
            "MediaPipe HandLandmarker initialized (model=%s, max_hands=%d)",
            resolved_path,
            num_hands,
        )

    @property
    def is_closed(self) -> bool:
        """Whether the underlying MediaPipe task has been closed."""
        return self._closed

    def detect_all(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        roi_percentage: float = 1.0,
    ) -> tuple[HandObservation, ...]:
        """Detect all hands, sorted by confidence (primary first)."""
        import numpy as np

        if self._closed or self._landmarker is None:
            raise RuntimeError("HandTracker is closed")
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty OpenCV image")
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        frame_height, frame_width = frame_bgr.shape[:2]
        roi_width = max(1, int(round(frame_width * roi_percentage)))
        roi_height = max(1, int(round(frame_height * roi_percentage)))
        x_offset = max(0, (frame_width - roi_width) // 2)
        y_offset = max(0, (frame_height - roi_height) // 2)
        detection_frame = frame_bgr[y_offset : y_offset + roi_height, x_offset : x_offset + roi_width]
        try:
            import cv2
            import mediapipe as mp

            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            raise RuntimeError(f"MediaPipe hand detection failed: {exc}") from exc
        finally:
            self._last_timestamp_ms = timestamp_ms
        observations: list[HandObservation] = []
        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            landmarks = tuple(
                _map_landmark_to_full_frame(lm, x_offset, y_offset, roi_width, roi_height, frame_width, frame_height)
                for lm in hand_lms
            )
            handedness = "Unknown"
            confidence = 0.0
            if result.handedness and hand_idx < len(result.handedness):
                cats = result.handedness[hand_idx]
                if cats:
                    cat = cats[0]
                    handedness = cat.category_name or cat.display_name or f"Hand {hand_idx}"
                    confidence = float(cat.score or 0.0)
            observations.append(HandObservation(landmarks, handedness, confidence))
        observations.sort(key=lambda o: o.confidence, reverse=True)
        return tuple(observations)

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        roi_percentage: float = 1.0,
    ) -> HandObservation | None:
        """Detect a hand in one BGR video frame.

        Args:
            frame_bgr: OpenCV BGR image.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.
            roi_percentage: Fraction of the centered frame sent to MediaPipe.
                Results are transformed back to full-frame normalized
                coordinates, so callers can always use the returned landmarks
                against the original frame.

        Returns:
            The first detected hand, or ``None`` when no hand is visible.

        Raises:
            RuntimeError: If the tracker is closed or MediaPipe inference fails.
            ValueError: If the frame is empty or an argument is invalid.
        """
        if self._closed or self._landmarker is None:
            raise RuntimeError("HandTracker is closed")
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty OpenCV image")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a three-channel BGR image")
        if not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a non-negative integer")
        if not 0.0 < roi_percentage <= 1.0:
            raise ValueError("roi_percentage must be in (0, 1]")
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        frame_height, frame_width = frame_bgr.shape[:2]
        roi_width = max(1, int(round(frame_width * roi_percentage)))
        roi_height = max(1, int(round(frame_height * roi_percentage)))
        x_offset = max(0, (frame_width - roi_width) // 2)
        y_offset = max(0, (frame_height - roi_height) // 2)
        detection_frame = frame_bgr[
            y_offset : y_offset + roi_height,
            x_offset : x_offset + roi_width,
        ]

        try:
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            raise RuntimeError(f"MediaPipe hand detection failed: {exc}") from exc
        finally:
            self._last_timestamp_ms = timestamp_ms

        if not result.hand_landmarks:
            return None

        landmarks = tuple(
            _map_landmark_to_full_frame(
                landmark,
                x_offset,
                y_offset,
                roi_width,
                roi_height,
                frame_width,
                frame_height,
            )
            for landmark in result.hand_landmarks[0]
        )
        handedness = "Unknown"
        confidence = 0.0
        if result.handedness:
            categories = result.handedness[0]
            if categories:
                category = categories[0]
                handedness = (
                    category.category_name
                    or category.display_name
                    or f"Hand {getattr(category, 'index', 0)}"
                )
                confidence = float(category.score or 0.0)
        return HandObservation(
            landmarks=landmarks,
            handedness=handedness,
            confidence=confidence,
        )

    def close(self) -> None:
        """Close the MediaPipe task; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        LOGGER.debug("MediaPipe HandLandmarker closed")

    def __enter__(self) -> "HandTracker":
        """Enter the tracker context."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the tracker when leaving its context."""
        self.close()


@dataclass(frozen=True)
class ProcessedFrame:
    """Result produced by the latest-frame processing worker."""

    frame_number: int
    timestamp_ms: int
    observation: HandObservation | None
    error: str | None = None
    observations: tuple[HandObservation, ...] = ()


class LatestFrameProcessor:
    """Run HandLandmarker in a worker while keeping only the newest frame.

    The queue is deliberately bounded.  When inference falls behind capture,
    the oldest pending frame is discarded instead of building latency.  The
    worker is the sole caller of ``HandTracker.detect`` while this class is
    active, so MediaPipe's native task is never used concurrently.
    """

    def __init__(
        self,
        tracker: HandTracker,
        roi_percentage: float = 1.0,
        queue_size: int = 1,
    ) -> None:
        """Start a latest-frame worker for ``tracker``."""
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if not 0.0 < roi_percentage <= 1.0:
            raise ValueError("roi_percentage must be in (0, 1]")
        self.tracker = tracker
        self.roi_percentage = roi_percentage
        self._queue: queue.Queue[
            tuple[int, np.ndarray, int] | None
        ] = queue.Queue(maxsize=queue_size)
        self._latest: ProcessedFrame | None = None
        self._latest_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="hand-landmarker-worker",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Consume frames until the stop sentinel is received."""
        while True:
            item = self._queue.get()
            try:
                if item is None or self._stop_event.is_set():
                    return
                frame_number, frame, timestamp_ms = item
                try:
                    observations = self.tracker.detect_all(
                        frame,
                        timestamp_ms,
                        roi_percentage=self.roi_percentage,
                    )
                    observation = observations[0] if observations else None
                    processed = ProcessedFrame(
                        frame_number, timestamp_ms, observation,
                        None, observations,
                    )
                except Exception as exc:  # Keep capture alive on worker errors.
                    LOGGER.exception("Asynchronous hand detection failed")
                    processed = ProcessedFrame(
                        frame_number, timestamp_ms, None, str(exc), ()
                    )
                with self._latest_lock:
                    self._latest = processed
            finally:
                self._queue.task_done()

    def submit(
        self, frame: np.ndarray, frame_number: int, timestamp_ms: int
    ) -> bool:
        """Submit a frame, replacing an older pending frame when necessary."""
        with self._lifecycle_lock:
            if self._stop_event.is_set():
                return False
            item = (frame_number, frame.copy(), timestamp_ms)
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                    return True
                except queue.Full:
                    return False

    def take_latest(self) -> ProcessedFrame | None:
        """Return and clear the newest completed result, if one exists."""
        with self._latest_lock:
            result = self._latest
            self._latest = None
        return result

    def close(self) -> None:
        """Stop the worker and wait until native inference has returned."""
        with self._lifecycle_lock:
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
        # The tracker must not be closed while the worker is inside MediaPipe.
        self._thread.join()


def _landmark_pixel(
    landmark: NormalizedLandmark, width: int, height: int
) -> tuple[int, int] | None:
    """Convert a normalized landmark to a bounded pixel coordinate."""
    if landmark.x is None or landmark.y is None:
        return None
    x = int(round(float(landmark.x) * (width - 1)))
    y = int(round(float(landmark.y) * (height - 1)))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def draw_hand_landmarks(
    frame: np.ndarray,
    landmarks: Sequence[NormalizedLandmark],
    landmark_color: tuple[int, int, int] = (0, 255, 0),
    connection_color: tuple[int, int, int] = (0, 200, 255),
    radius: int = 4,
) -> None:
    """Draw MediaPipe hand landmarks and all standard hand connections.

    Args:
        frame: BGR frame to annotate in place.
        landmarks: Twenty-one normalized MediaPipe landmarks.
        landmark_color: BGR color for landmark points.
        connection_color: BGR color for connection lines.
        radius: Landmark circle radius in pixels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or len(landmarks) < 21:
        return
    height, width = frame.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for index, landmark in enumerate(landmarks):
        if (
            landmark.visibility is not None
            and landmark.visibility < 0.5
        ) or (
            landmark.presence is not None
            and landmark.presence < 0.5
        ):
            continue
        point = _landmark_pixel(landmark, width, height)
        if point is not None:
            points[index] = point

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = getattr(connection, "start", None)
        end = getattr(connection, "end", None)
        if start is None or end is None:
            # Defensive compatibility with APIs exposing tuples rather than
            # Connection dataclasses.
            try:
                start, end = connection  # type: ignore[misc]
            except (TypeError, ValueError):
                continue
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], connection_color, 2)

    for index, point in points.items():
        point_radius = radius + (1 if index in (4, 8) else 0)
        cv2.circle(frame, point, point_radius, (255, 255, 255), -1)
        cv2.circle(frame, point, max(1, point_radius - 1), landmark_color, -1)
