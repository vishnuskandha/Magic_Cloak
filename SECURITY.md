# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please do not open a
public issue. Report it privately by opening a GitHub Security Advisory at:

https://github.com/vishnuskandha/Magic_Cloak/security/advisories

Include a description, steps to reproduce, and the impact. You should receive
an acknowledgment within 3 business days.

## Privacy and Webcam Data

- Magic Cloak is a **local desktop application**. All webcam frames are
  processed in memory on your machine; nothing is transmitted over the network.
- Saved frames (e.g. `cloak_<timestamp>.png` from `magic.py`) stay on your
  local disk. Share or upload them only if you are comfortable doing so.
- Review the code before running it with third-party camera software.

## Dependency Management

Keep dependencies up to date:

```bash
pip install --upgrade -r requirements.txt
```

If you spot a CVE affecting OpenCV or NumPy, update the pins in
`requirements.txt` and open a pull request.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| main    | :white_check_mark: |

## Security Best Practices for Contributors

- Run `python -m py_compile` on all modified `.py` files before committing.
- Do not commit webcam captures, screenshots, or other media files (the
  `.gitignore` already excludes `*.png`, `*.jpg`, `*.mp4`, `*.avi`, and
  `cloak_*.png`).
- Do not introduce network calls, telemetry, or hidden data collection.
