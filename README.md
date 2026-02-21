# Smart CrackSense

![Crack Detection](https://img.shields.io/badge/Crack%20Detection-blue)
![Image Processing](https://img.shields.io/badge/Image%20Processing-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-green)

Smart CrackSense is an open research codebase for automated crack detection and severity estimation on structural images. It combines image enhancement, segmentation, feature extraction, and classical machine learning to provide accurate detection and visual feedback for inspection workflows.

**Highlights**
- Color- and contrast-aware enhancement (CLAHE, denoising, sharpening)
- Multi-step segmentation (thresholding, edge detection, morphology)
- Geometric & texture feature extraction (length, width, orientation, GLCM)
- Trainable ML pipeline for severity classification (scripts + experiments storage)

## Quick Start

1. Clone the repository and change directory:

```bash
git clone https://github.com/your-username/Smart-CrackSense.git
cd "Smart CrackSense"
```

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS / Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Example runs:

- Build or prepare dataset:

```bash
python build_dataset.py
```

- Extract features for ML training:

```bash
python extract_dataset_features.py
```

- Train a ML model (saves outputs under `experiments/model_outputs`):

```bash
python src/models/train_ml.py
```

- Run main application (set `MODE` in `app.py` to `INFER` or `TRAIN`):

```bash
python app.py
```

## Repository Layout

- `app.py` — main entry for inference/train switch
- `build_dataset.py`, `extract_dataset_features.py` — data preparation
- `requirements.txt` — Python dependencies
- `dataset/` — raw and processed images
- `experiments/` — saved feature vectors, models, metrics, plots
- `src/`
  - `enhancement/` — CLAHE, grayscale, denoising, sharpening
  - `segmentation/` — edge detection, thresholding, morphological ops
  - `feature_extraction/` — length, width, density, orientation, texture
  - `models/` — training, evaluation, severity index utilities
  - `utils/` — image loading and visualization helpers

Use these scripts and modules as building blocks for experiments and integration into inspection pipelines.

## Usage Notes
- Default dataset: `SDNET2018/` (project includes crack/non_crack folders)
- Feature vectors are stored in `experiments/feature_vectors/X.npy` and `y.npy`
- Model outputs, performance metrics and plots are under `experiments/` for reproducibility

## Development & Contribution
- Create a feature branch for changes: `git checkout -b feat/your-change`
- Run unit checks on modified modules (add tests if possible)
- Open a PR with a clear description and reproducible steps

## Contact & Citation
If you use Smart CrackSense in research or production, please cite the repository and open an issue for feature requests or bugs.

## License
This project is distributed under the terms of the GNU General Public License v3.0. See the `LICENSE` file for details.

---

If you'd like, I can also:
- Add a short example notebook that walks through an end-to-end inference example
- Add GitHub Actions to run linting/tests on PRs

Let me know which of those you'd like next.
