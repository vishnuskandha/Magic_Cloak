# Magic Cloak

[![CI](https://github.com/vishnuskandha/Magic_Cloak/actions/workflows/ci.yml/badge.svg)](https://github.com/vishnuskandha/Magic_Cloak/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)

A real-time "invisible cloak" effect built with OpenCV. The program detects a
brightly colored cloth in the webcam feed and replaces it with the captured
background, making the person wearing it appear invisible.

## Features

Two programs cover different levels of use:

### 1. Simple Invisible Cloak (`invisible_cloak.py`)

- Captures a clean background (30 frames) at startup.
- Detects bright red and swaps it for the background in real time.
- Morphological cleanup for a stable mask.
- Press **ESC** to exit.

### 2. Advanced Magic Cloak (`magic.py`)

- Live **controls panel** with trackbars for HSV tuning.
- **8 color presets**: Red, Green, Blue, Yellow, Orange, Purple, Cyan, Magenta.
- **Color picker** - click on your cloak in the main window to auto-detect its color.
- **Background modes**: adaptive running-average background and a median static background capture.
- **Face and skin preservation** to avoid hiding your face.
- **Keyboard shortcuts**:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset adaptive background |
| `b` | Toggle background learning freeze |
| `g` | Grab a clean static background quickly |
| `G` | Enter static median capture mode (empty the scene first) |
| `u` | Use captured static background |
| `y` | Use adaptive background |
| `s` | Save current frame as `cloak_<timestamp>.png` |
| `0`-`7` | Select color preset |
| `m` | Toggle manual HSV mode |
| `f` | Toggle face preservation |
| `k` | Toggle skin preservation |
| `c` | Toggle color picker (click on cloak) |

### Common

- Red HSV ranges are tuned (high saturation/value) to avoid false positives on skin tones.
- Cross-platform (Windows, macOS, Linux).

## Requirements

- Python 3.7+
- OpenCV (`opencv-python>=4.5.0`)
- NumPy (`numpy>=1.19.0`)
- A webcam

## Quick Start

### Windows (one-click)

1. Double-click `setup.bat` to install the dependencies (first time only).
2. Double-click `run.bat`, choose option **1** (Simple) or **2** (Advanced).

### Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simple version
python invisible_cloak.py

# Or run the advanced version
python magic.py
```

## How to Use

1. Run one of the scripts - the camera window opens.
2. Step out of frame so the program captures a clean background.
3. Put on a bright red cloth (simple version) - or select/click a color (advanced version).
4. Step back into frame - the cloth disappears.
5. Exit with **ESC** (simple) or **q** (advanced).

## How It Works

1. **Background capture** - records the empty scene (static frame or running average).
2. **Color detection** - converts to HSV and thresholds the target color.
3. **Mask cleanup** - morphological open/close and temporal smoothing remove noise.
4. **Blending** - cloak-colored pixels are replaced by background pixels, frame by frame.

### HSV ranges (simple red detection)

```python
Lower Red 1: [0, 150, 100]     # hue 0-10, high saturation/value
Upper Red 1: [10, 255, 255]
Lower Red 2: [170, 150, 100]   # hue 170-180 (red wraps around hue)
Upper Red 2: [179, 255, 255]
```

## Project Structure

```
Magic_Cloak/
├── invisible_cloak.py    # Simple cloak (ESC to exit)
├── magic.py              # Advanced cloak with controls panel
├── setup.bat             # Windows dependency installer
├── run.bat               # Windows launcher menu
├── requirements.txt      # Python dependencies
├── CHANGELOG.md          # Release history
└── LICENSE               # MIT License
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Face gets hidden | Use a brighter cloth, improve lighting, enable face/skin preservation in `magic.py` |
| Cloak not detected | Check that the cloth is bright; adjust HSV or use the color picker |
| Camera not working | Check camera permissions and the device index (line 318 of `magic.py`) |
| Flickering effect | Improve lighting, use a solid background, stay still during capture |

## Security

This is a local desktop application - webcam frames are processed in memory and
never transmitted anywhere. See [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE). Copyright (c) 2025 Vishnu Skandha.
