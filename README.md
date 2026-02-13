# Smart CrackSense

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-GNU_3.0-green.svg)
![Tech](https://img.shields.io/badge/tech-Python_%7C_OpenCV_%7C_Scikit--Learn-orange.svg)

**Smart CrackSense** is an AI-powered structural health monitoring tool designed for precise crack detection and severity assessment. Leveraging advanced image processing techniques and machine learning, it automates the analysis of structural images to identify potential hazards efficiently.

This project combines robust image enhancement algorithms with feature extraction pipelines to feed a Random Forest classifier, enabling accurate categorization of crack severity levels.

---

## 🚀 Key Features

- **Advanced Image Enhancement**: Utilizes CLAHE, median filtering, and sharpening to improve image quality for analysis.
- **Automated Segmentation**: Employs adaptive thresholding, Canny edge detection, and morphological operations to isolate cracks.
- **Comprehensive Feature Extraction**: Calculates geometric features (length, width, orientation) and texture properties (GLCM contrast, homogeneity).
- **Severity Classification**: Machine learning model (Random Forest) classifies cracks into severity levels: **Non-Crack/Minor**, **Crack**, and **Severe Crack**.
- **Visual Feedback**: Generates and displays overlays of detected cracks on original images for easy verification.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Smart-CrackSense.git
    cd Smart-CrackSense
    ```

2.  **Set up the environment**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

 The application supports two modes: `TRAIN` and `INFER`. Configure the mode in `app.py`.

### Inference Mode (Default)
To analyze a single image:
1.  Ensure `MODE = "INFER"` in `app.py`.
2.  Set the `IMAGE_PATH` variable to your target image.
3.  Run the application:
    ```bash
    python app.py
    ```
    The system will display the processed image, the crack mask, and the overlay, along with the predicted severity.

### Training Mode
To retrain the model:
1.  Ensure you have precomputed feature vectors in `experiments/feature_vectors/` (`X.npy` and `y.npy`).
2.  Set `MODE = "TRAIN"` in `app.py`.
3.  Run the application to train and save the new model to `svm_model.pkl`.

## 📂 Project Structure

```
Smart-CrackSense/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── enhancement/        # Image preprocessing modules
│   ├── segmentation/       # Crack detection & masking
│   ├── feature_extraction/ # Geometric & texture analysis
│   ├── models/             # ML model training & inference
│   └── utils/              # Helper functions (loading, saving, plotting)
├── experiments/            # Intermediate outputs and model files
└── dataset/                # Input images for testing
```

## 🧩 Technologies Used

- **OpenCV**: Computer vision operations and image processing.
- **Scikit-Image**: Advanced image analysis.
- **Scikit-Learn**: Machine learning (Random Forest Classifier).
- **NumPy**: Numerical computing and array manipulation.
- **Matplotlib**: Visualization.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
