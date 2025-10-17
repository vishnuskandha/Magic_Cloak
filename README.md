# Invisible Cloak Effect

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Harry Potter-style invisible cloak effect using computer vision and OpenCV. This project creates a real-time invisibility effect by detecting red-colored objects and replacing them with the background.

## 🎬 Demo

![Invisible Cloak Demo](demo.gif)
*Put on a red cloak and become invisible!*

## Features

### **Two Magical Experiences:**
1. **Simple Invisible Cloak** (`invisible_cloak.py`)
   - Basic invisibility effect
   - Easy to use - just run and wear red!
   - Perfect for beginners
   
2. **Advanced Magic Cloak** (`magic.py`)
   - Professional controls panel
   - Multiple color presets (Red, Green, Blue, etc.)
   - Real-time HSV adjustments
   - Color picker tool
   - Background capture modes

### ✨ **Common Features:**
- **Real-time invisibility effect** using webcam
- **Optimized color detection** to avoid false positives with skin tones
- **Clean, professional code structure** with proper error handling
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **One-click launcher** with automatic dependency management

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Webcam/Camera

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishnuskandha/magic_cloak.git
   cd magic-cloak
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Run the program:**
   ```bash
   python invisible_cloak.py
   ```

## How to Use

1. **Run the script** - The program will initialize your camera
2. **Step out of frame** - Let the program capture a clean background (30 frames)
3. **Wear something red** - Put on a bright red cloth, shirt, or towel
4. **Step back into frame** - Watch yourself become invisible!
5. **Press ESC** to exit

## How It Works

### The Science Behind the Magic

1. **Background Capture**: Records a clean background without any objects
2. **Color Detection**: Uses HSV color space to detect bright red colors
3. **Mask Creation**: Creates masks to separate red objects from the rest
4. **Image Blending**: Replaces red areas with the corresponding background pixels
5. **Real-time Processing**: Applies the effect frame by frame

### Technical Details

```python
# HSV Color Ranges (optimized to avoid skin detection)
Lower Red 1: [0, 150, 100]   # Hue: 0-10°, High Saturation & Value
Upper Red 1: [10, 255, 255]
Lower Red 2: [170, 150, 100] # Hue: 170-180°, High Saturation & Value  
Upper Red 2: [180, 255, 255]
```

## 🎨 Customization

### Adjust Color Detection

Edit the HSV ranges in the code to detect different colors:

```python
# For blue cloak
lower_blue = np.array([100, 150, 100])
upper_blue = np.array([130, 255, 255])
```

### Modify Morphological Operations

Adjust the kernel size and iterations for different cloth textures:

```python
kernel = np.ones((5, 5), np.uint8)  # Larger kernel for smoother results
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **Face gets hidden** | Use brighter red cloth, ensure good lighting |
| **Cloak not detected** | Check if red is bright enough, adjust HSV ranges |
| **Camera not working** | Ensure camera permissions, check camera index |
| **Flickering effect** | Improve lighting conditions, use solid colored background |

### Tips for Best Results

- **Use bright, vibrant red cloth** (not dark or maroon)
- **Ensure good lighting** - avoid shadows
- **Use a plain background** for initial capture
- **Stay still** during background capture
- **Avoid red makeup or accessories** while testing

## Project Structure

```
magic-cloak/
├── invisible_cloak.py    # Main application
├── README.md            # This file
├── requirements.txt     # Dependencies
├── LICENSE             # MIT License
└── demo.gif           # Demo animation (add your own)
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit your changes**: `git commit -am 'Add some feature'`
4. **Push to the branch**: `git push origin feature-name`
5. **Submit a pull request**

### Ideas for Contributions

- [ ] Add support for multiple colors
- [ ] Implement GUI controls for HSV adjustment
- [ ] Add video recording functionality
- [ ] Create mobile app version
- [ ] Add different magical effects

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Harry Potter's Invisibility Cloak
- Built using OpenCV computer vision library
- Thanks to the open-source community

## Author

**Vishnu Skandha**
- GitHub: [@vishnuskandha](https://github.com/vishnuskandha)
- Project: [Magic Cloak Effect](https://github.com/vishnuskandha/magic-cloak)

---

**Star this repository** if you found it helpful!

**Report bugs** in the [Issues](https://github.com/vishnuskandha/magic-cloak/issues) section

**Suggest features** or improvements
