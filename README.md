# Phishing Detection AI

A Python-based project for detecting phishing content (emails/URLs) using machine learning models. This repository provides:

- 📌 **Dataset references** (CSV files under `data/`) for phishing and spam classification
- 🧠 **Model training & inference** (intended workflow) using `scikit-learn`
- 🚀 **FastAPI-based inference server** (replaces Flask)

---

## 🚀 Quick Start (Run the API)

### 1) Create & activate a virtual environment

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Train and compare multiple models (required for `/predict`)

The repository does **not** include a pre-trained model. Use the included training script to train from scratch with both datasets in `data/`:

- `data/phishing_email.csv` (uses `text_combined` + `label`)
- `data/PhiUSIIL_Phishing_URL_Dataset.csv` (uses `URL` + `label`)

Run:

```powershell
python train.py
```

This command will:
- Merge both datasets into one text-classification dataset
- Normalize labels to binary (`0 = legitimate`, `1 = phishing`)
- Split train/test with stratification
- Train and evaluate multiple TF-IDF pipelines:
  - `logistic_regression`
  - `linear_svc`
  - `multinomial_nb`
  - `sgd_classifier`
  - `passive_aggressive`
- Rank models by selection metric (default: `accuracy`)
- Save the best model to `saved_models/phishing_model.joblib`
- Save all model metrics to `saved_models/training_report.json`

Optional settings:

```powershell
python train.py --selection-metric accuracy
python train.py --models logistic_regression,linear_svc,multinomial_nb
python train.py --email-data data/phishing_email.csv --url-data data/PhiUSIIL_Phishing_URL_Dataset.csv --model-out saved_models/phishing_model.joblib --report-out saved_models/training_report.json
```

Available `--selection-metric` values:
- `accuracy`
- `phishing_f1`
- `macro_f1`
- `weighted_f1`

### 3.1) Train in Jupyter Notebook

Open `model_comparison.ipynb` and run cells from top to bottom.

Notebook output artifacts:
- `saved_models/phishing_model.joblib`
- `saved_models/training_report.json`

### 4) Start the FastAPI server

```powershell
uvicorn main:app --reload
```

Then open: http://127.0.0.1:8000/docs for interactive API docs.

---

## 🧩 Project Structure

```
.
├── api/                 # (Optional) API modules (currently empty)
├── data/                # Sample datasets (CSV files)
├── saved_models/        # Where trained models are stored (model file is expected here)
├── requirements.txt     # Python dependencies
├── main.py              # FastAPI app entrypoint
└── utils/
    └── model_utils.py   # Model loading + prediction helpers
```

---

## 🧠 API Reference (FastAPI)

### `GET /health`

Check that the service is running.

**Response**
- `200 OK` - `{ "status": "ok" }`

### `POST /predict`

Perform a phishing prediction.

**Request Body** (JSON)
```json
{
  "text": "Your message or URL to classify",
  "source": "optional source label (e.g. 'email', 'url')"
}
```

**Response**
```json
{
  "label": "phishing" | "legitimate",
  "score": 0.0,
  "details": { "source": "email" }
}
```

> ⚠️ If the model file is missing (`saved_models/phishing_model.joblib`), the server will return a 500 error on `/predict`.

---

## 🧪 Training Notes

Current default training setup in `train.py`:

- Feature extraction: `TfidfVectorizer` with uni+bi-grams
- Model family: multiple classifiers evaluated on the same split
- Label convention: `0 = legitimate`, `1 = phishing`

`training_report.json` now stores multi-model results including:
- selection metric and best model
- per-model metrics and timing
- ordered ranking list

If you want better performance later, you can try:

- URL-specific engineered features in addition to raw URL text
- Separate models for email and URL, then ensemble
- Hyperparameter search (`GridSearchCV`)

---

## ✅ Switching from Flask to FastAPI

As requested, this repository uses **FastAPI** (instead of Flask) for the inference API.

FastAPI provides:
- Automatic interactive docs (Swagger UI at `/docs`)
- High performance (Starlette + Uvicorn)
- Type-safe request/response models with Pydantic

---

## 📝 Notes & Next Steps

- Add real preprocessing + feature extraction pipelines under `processing/` or `features/`.
- Store/track training experiments (e.g., with `mlflow`, `weights & biases`, or simple notebook logs).
- Add unit tests for model loading + prediction.

---

If you want, I can also add a starter training script (e.g., `train.py`) that reads `data/`, trains a simple classifier, and saves it for inference.
