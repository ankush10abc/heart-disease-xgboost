import pickle
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Prediction Demo")

model = None
FEATURES = None

def load_model():
    global model, FEATURES
    candidates = [
        Path(__file__).resolve().parent / "model.pkl",  # same dir as app.py
        Path(os.getcwd()) / "model.pkl",                # working directory
        Path("/var/task/model.pkl"),                    # Vercel Lambda root
    ]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            model = bundle["model"]
            FEATURES = bundle["features"]
            print(f"Model loaded from: {path}")
            return
        except Exception as e:
            print(f"Failed {path}: {e}")
    print("ERROR: model.pkl not found")

load_model()


class PatientData(BaseModel):
    HighBP: int = Field(ge=0, le=1)
    HighChol: int = Field(ge=0, le=1)
    CholCheck: int = Field(ge=0, le=1)
    BMI: float = Field(gt=0, le=100)
    Smoker: int = Field(ge=0, le=1)
    Stroke: int = Field(ge=0, le=1)
    Diabetes: int = Field(ge=0, le=2)
    PhysActivity: int = Field(ge=0, le=1)
    HvyAlcoholConsump: int = Field(ge=0, le=1)
    MentHlth: int = Field(ge=0, le=30)
    PhysHlth: int = Field(ge=0, le=30)
    Sex: int = Field(ge=0, le=1)
    Age: int = Field(ge=1, le=13)
    Education: int = Field(ge=1, le=6)
    Income: int = Field(ge=1, le=8)


FIELD_DEFS = [
    ("HighBP",            0,  1,  1,   "1"),
    ("HighChol",          0,  1,  1,   "1"),
    ("CholCheck",         0,  1,  1,   "1"),
    ("BMI",               10, 70, 30,  "0.1"),
    ("Smoker",            0,  1,  0,   "1"),
    ("Stroke",            0,  1,  0,   "1"),
    ("Diabetes",          0,  2,  0,   "1"),
    ("PhysActivity",      0,  1,  1,   "1"),
    ("HvyAlcoholConsump", 0,  1,  0,   "1"),
    ("MentHlth",          0,  30, 0,   "1"),
    ("PhysHlth",          0,  30, 0,   "1"),
    ("Sex",               0,  1,  0,   "1"),
    ("Age",               1,  13, 7,   "1"),
    ("Education",         1,  6,  4,   "1"),
    ("Income",            1,  8,  5,   "1"),
]


@app.get("/", response_class=HTMLResponse)
def home():
    field_names_js = str([f[0] for f in FIELD_DEFS])
    fields_html = "".join(
        '<label>' + name + '<input name="' + name + '" type="number" '
        'min="' + str(mn) + '" max="' + str(mx) + '" step="' + step + '" value="' + str(val) + '" required></label>'
        for name, mn, mx, val, step in FIELD_DEFS
    )

    html = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart Health ML Demo</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f3f6fb; margin: 0; padding: 24px; color: #172033; }
    main { max-width: 850px; margin: auto; background: white; padding: 28px; border-radius: 16px; box-shadow: 0 5px 24px #17203312; }
    h1 { color: #174ea6; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
    label { font-size: 14px; font-weight: bold; display: block; }
    input { display: block; box-sizing: border-box; width: 100%; margin-top: 7px; padding: 11px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }
    button { margin-top: 22px; width: 100%; padding: 14px; border: 0; border-radius: 8px; background: #174ea6; color: white; font-size: 16px; cursor: pointer; }
    .note { color: #596579; font-size: 13px; }
    /* Modal */
    #modalOverlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 100; align-items: center; justify-content: center; }
    #modalOverlay.show { display: flex; }
    #modalBox { background: white; border-radius: 16px; padding: 32px 28px 24px; max-width: 420px; width: 90%; box-shadow: 0 8px 40px rgba(0,0,0,0.18); text-align: center; }
    #modalIcon { font-size: 48px; margin-bottom: 12px; }
    #modalTitle { font-size: 20px; font-weight: bold; margin-bottom: 12px; }
    #modalBody { font-size: 15px; line-height: 1.7; margin-bottom: 24px; }
    #modalClose { padding: 10px 36px; border: 0; border-radius: 8px; font-size: 15px; cursor: pointer; color: white; }
  </style>
</head>
<body>
<main>
  <h1>Heart Health Prediction</h1>
  <p>Enter the coded features used by the training dataset.</p>
  <form id="patientForm">
    <div class="grid">FIELDS_PLACEHOLDER</div>
    <button type="submit">Run Prediction</button>
  </form>
  <p class="note">Educational ML demo only. This tool is not a medical diagnosis.</p>
</main>

<div id="modalOverlay">
  <div id="modalBox">
    <div id="modalIcon"></div>
    <div id="modalTitle"></div>
    <div id="modalBody"></div>
    <button id="modalClose" onclick="closeModal()">Close</button>
  </div>
</div>

<script>
  var fieldNames = FIELD_NAMES_PLACEHOLDER;

  function showModal(success, title, bodyHTML) {
    document.getElementById('modalIcon').textContent = success ? '\u2705' : '\u274C';
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalTitle').style.color = success ? '#16a34a' : '#dc2626';
    document.getElementById('modalBody').innerHTML = bodyHTML;
    document.getElementById('modalClose').style.background = success ? '#16a34a' : '#dc2626';
    document.getElementById('modalOverlay').classList.add('show');
  }

  function closeModal() {
    document.getElementById('modalOverlay').classList.remove('show');
  }

  document.getElementById('modalOverlay').addEventListener('click', function(e) {
    if (e.target === this) closeModal();
  });

  document.getElementById('patientForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var btn = event.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Predicting...';

    var data = {};
    for (var i = 0; i < fieldNames.length; i++) {
      var name = fieldNames[i];
      data[name] = Number(document.querySelector('[name="' + name + '"]').value);
    }

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    .then(function(response) {
      var status = response.status;
      return response.text().then(function(text) {
        return { status: status, text: text };
      });
    })
    .then(function(res) {
      btn.disabled = false;
      btn.textContent = 'Run Prediction';
      var output;
      try {
        output = JSON.parse(res.text);
      } catch(e) {
        showModal(false, 'Server Error', 'Could not parse response:<br>' + res.text.substring(0, 200));
        return;
      }
      if (res.status !== 200) {
        showModal(false, 'Prediction Failed', 'Error ' + res.status + ': ' + (output.detail || 'Unknown error'));
        return;
      }
      var label = output.prediction === 1 ? '\u2764\uFE0F Heart Disease Detected' : '\uD83D\uDC9A No Heart Disease Detected';
      var body = '<strong>' + label + '</strong><br><br>'
        + 'Probability No Heart Disease: <strong>' + (output.probability_class_0 * 100).toFixed(2) + '%</strong><br>'
        + 'Probability Heart Disease: <strong>' + (output.probability_class_1 * 100).toFixed(2) + '%</strong>';
      showModal(true, 'Prediction Result', body);
    })
    .catch(function(err) {
      btn.disabled = false;
      btn.textContent = 'Run Prediction';
      showModal(false, 'Network Error', err.message);
    });
  });
</script>
</body>
</html>
"""
    html = html.replace('FIELDS_PLACEHOLDER', fields_html)
    html = html.replace('FIELD_NAMES_PLACEHOLDER', field_names_js)
    return html


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/debug")
def debug():
    import sys
    cwd = os.getcwd()
    file_dir = str(Path(__file__).resolve().parent)
    files_cwd = os.listdir(cwd)
    files_dir = os.listdir(file_dir) if file_dir != cwd else "(same as cwd)"
    return {
        "cwd": cwd,
        "file_dir": file_dir,
        "files_in_cwd": files_cwd,
        "files_in_file_dir": files_dir,
        "model_loaded": model is not None,
        "features": FEATURES,
        "python": sys.version,
    }


@app.post("/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    values = patient.model_dump()
    input_df = pd.DataFrame([values], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    return {
        "prediction": prediction,
        "probability_class_0": float(probabilities[classes.index(0)]),
        "probability_class_1": float(probabilities[classes.index(1)])
    }
