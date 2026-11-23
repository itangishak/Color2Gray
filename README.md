# Color-to-Grayscale Conversion

Implementation of Color2Gray (Gooch et al., SIGGRAPH 2005) extended with a saliency-aware boosting pipeline for isoluminant scenes (e.g., Ishihara plates).

## Methods

- **Standard**: L* channel only.
- **Vanilla**: Gooch 2005 Poisson-based perceptual grayscale.
- **Enhanced**: Vanilla + multi-scale guided chroma details, phase-inhibition, competitive saliency, and haze/noise gating. Increases contrast for isoluminant content while keeping natural images stable.

## Performance

- Large images are auto-downsampled to 800 px (all methods) or 1024 px (single method).
- Typical memory usage: 200–300 MB. Processing time: 10–30 s.

## Setup

```bash
python -m venv color2gray
source color2gray/bin/activate  # Windows: color2gray\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python app.py
# Open http://127.0.0.1:5001
```

Upload an image and select a method. Parameters:

- `theta`: Gooch solver trade-off (vanilla and enhanced).
- `alpha`: Chroma boost strength (enhanced only).

Results are saved to `static/results/`.

## Programmatic API

```python
from utils.color2gray_impl import color2gray_vanilla, color2gray_enhanced

g1 = color2gray_vanilla(rgb_img, theta=30.0)
g2 = color2gray_enhanced(rgb_img, theta=30.0, alpha=4.0)
```

## Files

- `utils/color2gray_impl.py` — vanilla and enhanced implementations.
- `utils/guided_filter.py` — edge-aware guided filter.
- `app.py` — Flask app.

## Requirements

Python 3.9+, NumPy, SciPy, Pillow, Flask. See `requirements.txt`.
