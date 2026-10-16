# Virtual-Mouse — Gap to God-Level + Plan

> تاریخ: 2026-10-16 — شاخه: `a08ebcd Refactor into modular architecture`
> فایل‌های مرجع: `main.py:1` (999 خط)، `gesture_recognizer.py:1` (648)، `hand_tracker.py:1` (545)، `mouse_controller.py:131` (311)، `config.py:309` (437)، `config.json:1` (74)

## 1. وضعیت فعلی (نمره 6/10)

**خوب:**
- کانفیگ تایپ‌دار + validation در `config.py:309`، `self-test` بدون دوربین در `main.py:501` (1.06s پاس شد)
- `LatestFrameProcessor` با صف 1 فریمی `hand_tracker.py:370`، `ClickDebouncer` با stable+cooldown `mouse_controller.py:25`
- دانلود اتمیک مدل `hand_tracker.py:81`، ROI mapping `hand_tracker.py:59`

**ضعیف (دیده شده در ران):**
- CPU ~258٪، FPS نوسانی 31-69، `Killed 137` بعد 1600 فریم، ارور Qt `QFont::setPointSizeF`
- دوربین `select() timeout` قبل از اتصال وب‌کم، بدون reconnect `main.py:180`
- `run_application:617` ~350 خط god-function، OSD + inference + mouse همه در یک thread اصلی

## 2. فاصله تا سطح خدا (God = <30ms، >99٪ دقت، <1 false-click/hr، <15٪ CPU)

### A. دقت ژست (بزرگ‌ترین فاصله)
- `gesture_recognizer.py:45` فقط 6 ژست هندسی با آستانه ثابت: `pinch 0.055:477`، `fist 0.52:459`، `peace sep 0.035:489`
- اولویت `fist>pinch>peace>scroll>pointing:560` در `classify_gesture:539` باعث بلعیده شدن pinch شل توسط fist می‌شود
- کلیک چپ فقط از `TapDetector:93` (`velocity>=0.05, drop>=0.02, window 3-5f`) — اگر لحظه‌ای از pinch رد شود `suppress_tap:723` history را reset می‌کند = کلیک گمشده
- ندارد: drag/hold (فقط در `mouse_legacy.py:77` بود، در `main.py` حذف شده)، dwell-click، hold طولانی، دو دست (`num_hands=1` در `config.json:51`)، کالیبراسیون pinch/fist/peace (فقط `--calibrate-tap:124` دستی)

### B. لتنسی/پرفورمنس
- `HandTracker.detect:252` سنکرون + `frame.copy() + cvtColor + ascontiguous` هر فریم 2x کپی
- `PyAutoGUI moveTo` هر فریم `main.py:750` ~5-15ms X11 IPC، با `PAUSE=0` هم کند است؛ Wayland خراب است `mouse_controller.py:156`
- smoothing واحد `current+(target-current)/5.0:224` — یا لگ یا jitter
- پیش‌فرض سنگین‌ترین حالت: `process_every_n_frames=1, roi=0.7` در `config.py:256`؛ بدون FPS cap، busy-loop 100٪ یک هسته
- `map_to_screen:195` فقط primary مانیتور `pyautogui.size()`، بدون multi-monitor/DPI

### C. قابلیت اطمینان/امنیت
- بعد `failed_reads>=5:672` خروج کامل، بدون backoff/reconnect
- دانلود مدل بدون checksum/signature `hand_tracker.py:81` (فقط `MIN_SIZE=1MB`)، `http://` قبول می‌کند `config.py:301`
- لاگ فقط stderr `main.py:88`، بدون file/rotation (با اینکه `*.log` در `.gitignore:10` هست)
- `opencv-python + opencv-contrib-python` همزمان در `requirements.txt:1` conflict دارند؛ همه `>=` بدون lock

### D. تست/دیپلوی/UX
- بدون `tests/`, `pytest`, `coverage`, `CI` — فقط synthetic `--self-test`
- بدون video-regression، بدون بنچ p95، بدون mock برای PyAutoGUI
- OSD ابتدایی OpenCV `main.py:259` (10 پارامتر)، بدون overlay شفاف/GUI تنظیمات
- بدون `Dockerfile`, `systemd`, installer؛ README پیش‌نیاز سیستمی (libGL, X11, v4l2) ندارد

## 3. پلن P0-P3

### P0 — پایداری (1-2 روز) — معیار: 600 فریم بدون Kill، reconnect کار کند
- [ ] ریفکتور `run_application:617` به `camera.py + osd.py + app.py` (god-function را بشکن)
- [ ] reconnect دوربین با backoff به‌جای `CameraError` فوری؛ چک return مقدار `capture.set()`
- [ ] fix QFont (فونت Qt یا switch به fontconfig)، `thread.join(timeout)` در `hand_tracker.py:482`
- [ ] `requirements.txt` را pin کن + `opencv-contrib` تنها + `requirements-lock.txt` + `pytest/ruff` به dev-extra
- [ ] لاگ فایل چرخشی + checksum SHA256 مدل + block کردن `http://`
- [ ] قبول: `main.py --no-display --max-frames 600` دو بار پشت سر هم EXIT 0، مموری flat

### P1 — دقت + لتنسی (3-5 روز) — معیار: p95<33ms، false-click نصف شود
- [ ] فیلتر OneEuro/Kalman روی لندمارک قبل از `move_to_normalized` (نه فقط موس)
- [ ] کالیبراسیون 10ثانیه‌ای per-user برای pinch/fist/peace (مثل tap) + ذخیره در `config.json` + کلید `c` برای شروع
- [ ] hysteresis برای scroll `deadzone 0.035` + رفع dead-zone 5 فریمی `tap_conflict_until:739`
- [ ] hold-to-drag: `pinch-hold >300ms = mouseDown` (برگرداندن `left_dragging` از legacy به شکل debounced)
- [ ] `process_every_n_frames=2`, `roi=0.5`, رزولوشن 480p پیش‌فرض + FPS cap + sleep در لوپ
- [ ] قبول: بنچ 300 فریمی روی ویدیو ثابت، گزارش `FPS p50/p95 + clicks`

### P2 — موس مدرن + UX (3-4 روز) — معیار: Wayland کار کند، مولتی‌مانیتور
- [ ] backend موس: `evdev/uinput (linux) > xdotool > pyautogui` با auto-detect + enumerate مانیتورها
- [ ] dwell-click (سکون 600ms در شعاع 12px) + تنظیم در `ClickSettings`
- [ ] OSD جدید: پنل transparent، نمودار velocity/drop برای tap، نمایش `UNKNOWN` دلیل‌دار
- [ ] قبول: تست روی X11 + Wayland، حرکت پیوسته روی 2 مانیتور

### P3 — خدا (1-2 هفته) — معیار: >99٪ روی ست ویدیویی، <15٪ CPU
- [ ] classifier سبک (logistic/MLP روی فیچرهای `finger_states + distances + pinch`) جایگزین rule ثابت، با دیتاست ویدیویی خودت
- [ ] دو دست: دست غالب موس + دست دوم شورتکات (scroll/alt-tab)
- [ ] `tests/` + `pytest + coverage>90` + video-regression + `GitHub Actions` + `Dockerfile`
- [ ] قبول: ست 5 ویدیویی (نور کم/زیاد، پوست مختلف، فاصله دور/نزدیک) با precision/recall گزارش‌شده

## 4. قدم بعدی پیشنهادی
شروع با P0-1 (ریفکتور + reconnect + pin requirements) چون Kill و god-function جلوی همه کارهای بعدی را می‌گیرد.
