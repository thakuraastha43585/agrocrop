# AgroVision‑AI

**Smart tomato disease detection web application** built with **Streamlit**, **TensorFlow/Keras**, and a **MobileNetV2** transfer‑learning model trained on the PlantVillage dataset.

---

## Demo

*(Add a GIF or screenshot of the app here.)*

---

## Installation (local)

```powershell
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/AgroVision-AI.git
cd AgroVision-AI

# Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# or `source .venv/bin/activate` on macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Usage

1. **Home** – Project description, objectives, and SDG relevance.
2. **Disease Detection** – Upload a tomato leaf image (JPG/PNG). The model predicts one of five classes and shows confidence.
3. **Model Performance** – Displays test accuracy, precision/recall/F1, confusion matrix, and training curves (read from `reports/evaluation_results.json`).
4. **About** – Explains MobileNetV2 transfer learning, dataset, preprocessing, and future work.

---

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (see next section).
2. Log in to **Streamlit Community Cloud** (https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**, select the repo, branch `main`, and entry point `app.py`.
4. Streamlit will automatically install the packages from `requirements.txt` and launch the app.
5. After a successful build, you will obtain a public URL like `https://yourusername-AgroVision-AI-xxxxxx.streamlit.app`.

---

## Model Details

- **Architecture**: MobileNetV2 (pre‑trained on ImageNet) + custom classification head for 5 tomato disease classes.
- **Input shape**: 224 × 224 × 3.
- **Training/validation split**: 70 % / 15 % / 15 % (stratified, fixed seed).
- **Test accuracy**: **90.77 %** on 1,137 unseen images.
- **Metrics**: Precision, recall, F1‑score (macro and weighted), confusion matrix (see `reports/figures/`).

---

## Repository Contents

```
AgroVision-AI/
│   app.py                 # Streamlit entry point
│   requirements.txt       # Minimal dependencies for deployment
│   README.md              # (this file)
│   .gitignore             # Excludes env, raw data, caches, etc.
│
├───models/                # Trained model (crop_disease_model.keras)
├───data/                  # *raw* dataset is ignored; only manifest files kept
├───src/                   # src/predict.py, src/data_loader.py, etc.
├───reports/               # Figures and evaluation_results.json
└───final_submission/      # SRS DOCX/PDF and PPTX for the college report
```

---

## License

This project is released under the **MIT License**. Feel free to fork, modify, and use it for educational purposes. Cite the original work if used in publications.

---

*Prepared by the AgroVision‑AI development team.*