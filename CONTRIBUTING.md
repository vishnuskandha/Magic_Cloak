# Contributing to Magic Cloak

Thanks for taking the time to contribute. Please read [README.md](README.md)
first so your changes stay consistent with the project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make your changes and verify them (see below).
4. Commit with a clear message and open a pull request against `main`.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Before Submitting

- Run the syntax check on every changed file:

  ```bash
  python -m py_compile magic.py invisible_cloak.py
  ```

- The CI workflow runs `pip install -r requirements.txt` and the same
  `py_compile` check on every push/PR.
- Do not add test code that requires a webcam to CI; the scripts open the
  camera only when run interactively.
- Do not commit media files (`*.png`, `*.jpg`, `*.mp4`, `*.avi`, `cloak_*.png`)
  or virtual environments.

## Style

- Follow the existing style: modular classes/functions, clear docstrings,
  descriptive variable names.
- Keep magic numbers as named constants or class defaults with comments.
- Keep both scripts consistent: if you change HSV defaults in `magic.py`,
  consider whether `invisible_cloak.py` needs the same change.

## Commit Messages

Use concise, descriptive messages, e.g.:

```
Add yellow color preset to advanced cloak
```

## Pull Request Checklist

- [ ] `py_compile` passes for all changed files
- [ ] `README.md` updated if behavior, controls, or requirements change
- [ ] `CHANGELOG.md` entry added for user-facing changes
- [ ] No media files, credentials, or virtual environments added

## Reporting Bugs

Open an issue with your OS, Python version, OpenCV version, and the full error
output. If it is security-related, follow [SECURITY.md](SECURITY.md).
