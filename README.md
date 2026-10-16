# 🖱️ Virtual Mouse Using Hand Gestures

A local, webcam-based virtual mouse built with Python 3.14, OpenCV, PyAutoGUI, and the modern MediaPipe Tasks `HandLandmarker` API.

The application tracks one hand in real time, moves the cursor with the index finger, and recognizes click and scroll gestures. Gesture edges are stabilized and rate-limited so holding a pose does not produce a stream of duplicate clicks.

> **Privacy:** Camera frames are processed locally. The application does not upload video or landmark data.

## ✨ Features

- **Modern MediaPipe Tasks API** using `mediapipe.tasks.python.vision.hand_landmarker.HandLandmarker` (MediaPipe 1.0+).
- **Index-finger cursor control** with exponential low-pass smoothing, a pixel dead zone, and configurable screen-region mapping.
- **Left click:** thumb and index fingertip pinch.
- **Right click:** all five fingertips close to the wrist (fist).
- **Double click:** index and middle finger peace sign.
- **Optional scroll:** thumb up/down with the other fingers curled.
- **Click debounce:** stable-frame confirmation, gesture edge detection, and a configurable cooldown (20 frames by default).
- **On-screen display:** FPS, current gesture, handedness, landmarks, standard hand connections, cooldown state, and click feedback rings.
- **Automatic model download:** the `.task` model is downloaded atomically on first launch if it is missing.
- **Runtime adjustments:** reload configuration, change smoothing, toggle overlays, and save settings without editing code.
- **Graceful shutdown and error handling:** press `q` or `Esc` to release the camera and MediaPipe resources.

## Demo

> **Demo GIF/video placeholder:** add a short capture showing pointing, pinch, fist, and peace gestures here.

![Demo placeholder](docs/demo.gif)

> **Screenshot placeholder:** replace this with a screenshot of the OSD and hand landmarks.

![Screenshot placeholder](docs/screenshot.png)

## 🧰 Requirements

- Python **3.14** (the project was validated against Python 3.14.0; Python 3.11+ is generally suitable for the same code).
- A webcam and a desktop session with a visible mouse/display.
- Internet access on the first launch so the MediaPipe model can be downloaded.

The dependency versions are intentionally minimum-version constrained in [`requirements.txt`](requirements.txt):

```text
mediapipe>=1.0.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
pyautogui>=0.9.54
numpy>=1.24.0
```

## ⚙️ Installation

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, either enable local script execution for the current user or invoke the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### macOS/Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The first launch downloads the model to:

```text
models/hand_landmarker.task
```

The source URL is:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

You can also provide a local model explicitly:

```powershell
python main.py --model C:\path\to\hand_landmarker.task
```

## 🖐️ Gesture guide

| Gesture | Action | Detection rule |
| --- | --- | --- |
| Index finger extended, other three fingers curled | Move cursor | Cursor follows the index fingertip; thumb position is ignored for movement |
| Thumb + index pinch | Left click | Thumb/index distance is at or below `pinch_threshold` |
| Fist | Right click | Every fingertip (thumb and four fingers) is closer to the wrist than `fist_tip_wrist_threshold` |
| Index + middle peace sign | Double click | Index and middle are extended, ring and pinky are curled, and fingertips are separated |
| Thumb up, other fingers curled | Scroll up | Optional scroll gesture |
| Thumb down, other fingers curled | Scroll down | Optional scroll gesture |

A gesture must be visible for the configured number of stable frames and must be released before it can fire again. The default 20-frame cooldown prevents click spam.

## ⌨️ Runtime controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit cleanly |
| `[` | Decrease smoothing factor by 0.5 (more responsive) |
| `]` | Increase smoothing factor by 0.5 (more stable) |
| `r` | Reload `config.json` |
| `s` | Save the current runtime settings to `config.json` |
| `h` | Show/hide gesture instructions |
| `d` | Show/hide hand landmarks |

`r` reloads live-adjustable gesture, mouse, click, and display settings. Camera capture mode, model path, and MediaPipe confidence settings take effect after restarting the application.

## 🧾 Configuration guide

All settings live in [`config.json`](config.json). Values are validated at startup and unknown keys are ignored with a warning.

### `camera`

- `index`: OpenCV camera index. Common values are `0` for the default webcam and `1` for a second camera.
- `width`, `height`, `fps`: Requested capture mode. The OSD uses the actual frames returned by the driver.
- `flip`: Mirror the preview and tracking input like a normal self-view.

### `mouse`

- `smoothing_factor`: Exponential low-pass factor. The update is `current + (target - current) / factor`; `7.0` is the default balance between latency and stability. The legacy spelling `smoothening` is also accepted in this object.
- `movement_scale`: Expands movement around the center of the selected region. Keep at `1.0` for direct mapping.
- `dead_zone_pixels`: Suppresses tiny model movements below this many desktop pixels.
- `screen_region`: Normalized screen rectangle (`x`, `y`, `width`, `height`). For example, the left half of the screen is `x=0`, `width=0.5`.

### `gestures`

- `pinch_threshold`: Maximum normalized thumb/index distance for a pinch.
- `fist_tip_wrist_threshold`: Maximum normalized fingertip-to-wrist distance for every finger in a fist.
- `finger_extension_ratio` and `finger_angle_threshold_degrees`: Finger extension geometry thresholds.
- `peace_min_separation`: Minimum index/middle fingertip separation for a peace sign.
- `scroll_enabled`, `scroll_deadzone`, `scroll_amount`, `scroll_cooldown_frames`: Optional scroll controls.

### `clicks`

- `cooldown_frames`: Minimum frames between emitted clicks.
- `stable_frames`: Frames a gesture must remain stable before it can click.
- `double_click_interval_frames`: Reserved for future gesture timing refinements; the peace gesture currently calls `doubleClick()` directly.
- `feedback_frames`: Duration of the colored click ring on the OSD.

### `detection`

- `num_hands`: Maximum hands passed to MediaPipe. The cursor uses the first detected hand.
- `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence`: MediaPipe confidence thresholds.

### `display`

- `window_name`: OpenCV preview window title.
- `show_landmarks`, `show_instructions`, `show_fps`: OSD visibility switches.

## 🏗️ Project structure

```text
Virtual-Mouse/
├── main.py                 # CLI, camera loop, OSD, and application lifecycle
├── hand_tracker.py         # MediaPipe Tasks model download and HandLandmarker wrapper
├── gesture_recognizer.py   # Landmark geometry and gesture classification
├── mouse_controller.py     # Screen mapping, smoothing, clicks, scroll, debounce
├── config.py               # Typed config loading and validation
├── config.json             # User-adjustable settings
├── mouse_legacy.py         # Preserved original implementation
├── requirements.txt
├── README.md
└── models/
    └── hand_landmarker.task
```

`mouse_legacy.py` is retained for reference only. The supported entry point is `main.py`; the legacy file intentionally contains the old `mp.solutions` code.

## 🧪 Validation

The following command does not require a physical camera. It validates configuration loading, the model, MediaPipe VIDEO-mode initialization, synthetic gesture cases, click debounce, and screen mapping:

```powershell
python main.py --self-test
```

A short camera/window smoke test can be run with:

```powershell
python main.py --max-frames 300
```

For a non-interactive capture smoke test (the process exits after the frame limit):

```powershell
python main.py --no-display --max-frames 30
```

## 🛠️ Troubleshooting

### The camera does not open

- Close other applications using the webcam.
- Try another index, such as `--camera 1`, or change `camera.index` in `config.json`.
- On Windows, check **Settings → Privacy & security → Camera** and allow desktop apps to access the camera.
- Confirm the camera is not in use by a browser or video-conferencing application.

### The model does not download

- Check the exact URL in `model.url` and your network/proxy settings.
- Download `hand_landmarker.task` manually into `models/` using the URL above, then rerun.
- Delete a truncated `models/hand_landmarker.task.part` file and retry.
- The downloader writes to a temporary file and only renames it after a complete response.

### `mediapipe` import fails

- Activate the intended Python 3.14 environment and reinstall requirements.
- Confirm the installed package is at least 1.0.0:

  ```powershell
  python -c "import mediapipe; print(mediapipe.__version__)"
  ```

- This project intentionally does **not** import `mp.solutions.hands`.

### PyAutoGUI cannot access the screen

PyAutoGUI needs a desktop session and a visible display. Run the program locally rather than through a headless service, remote terminal, or locked Windows session. On Linux, an X11/Wayland-compatible desktop backend may be required.

### Cursor is jittery or too slow

- Increase `mouse.smoothing_factor` or press `]`.
- Increase `mouse.dead_zone_pixels` slightly.
- Improve lighting and keep the hand fully inside the camera view.
- Reduce the requested camera resolution if inference is slow.

### Gestures trigger incorrectly

- Adjust `gestures.fist_tip_wrist_threshold` and `gestures.pinch_threshold` in small steps.
- Keep a consistent camera distance and lighting.
- Avoid making the thumb touch the index while forming a fist; fist has intentional priority.
- Use a larger `clicks.stable_frames` value when recognition is intermittent.

### FPS is below the target

- Close other camera/video applications.
- Lower `camera.width` and `camera.height` if the CPU is the bottleneck.
- Keep `detection.num_hands` at `1` unless multiple-hand support is needed.
- The OSD FPS value includes capture, inference, gesture processing, and drawing.

## 📄 License

This project is licensed under the terms of the [MIT license](LICENSE.md).
