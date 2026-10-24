import pickle
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, create_model

app = FastAPI(title="Heart Disease Prediction Demo")

model    = None
FEATURES = None
DROPPED  = None

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    global model, FEATURES, DROPPED
    candidates = [
        Path(__file__).resolve().parent / "model.pkl",
        Path(os.getcwd()) / "model.pkl",
        Path("/var/task/model.pkl"),
    ]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            model    = bundle["model"]
            FEATURES = bundle["features"]
            DROPPED  = bundle.get("dropped", [])
            print(f"Model loaded from: {path}")
            print(f"Selected features ({len(FEATURES)}): {FEATURES}")
            print(f"Dropped  features ({len(DROPPED)}): {DROPPED}")
            return
        except Exception as e:
            print(f"Failed {path}: {e}")
    print("ERROR: model.pkl not found")

load_model()

# ── Field metadata for ALL possible features ─────────────────────────────────
# (name, min, max, default, step)
ALL_FIELD_META = {
    "HighBP":            (0,  1,  1,   "1"),
    "HighChol":          (0,  1,  1,   "1"),
    "CholCheck":         (0,  1,  1,   "1"),
    "BMI":               (10, 70, 30,  "0.1"),
    "Smoker":            (0,  1,  0,   "1"),
    "Stroke":            (0,  1,  0,   "1"),
    "Diabetes":          (0,  2,  0,   "1"),
    "PhysActivity":      (0,  1,  1,   "1"),
    "HvyAlcoholConsump": (0,  1,  0,   "1"),
    "MentHlth":          (0,  30, 0,   "1"),
    "PhysHlth":          (0,  30, 0,   "1"),
    "Sex":               (0,  1,  0,   "1"),
    "Age":               (1,  13, 7,   "1"),
    "Education":         (1,  6,  4,   "1"),
    "Income":            (1,  8,  5,   "1"),
}

# ── Build Pydantic model dynamically from selected features only ──────────────
def build_patient_model(features):
    fields = {}
    for name in features:
        mn, mx, default, _ = ALL_FIELD_META[name]
        if name == "BMI":
            fields[name] = (float, Field(default=float(default), gt=0, le=mx))
        else:
            fields[name] = (int, Field(default=int(default), ge=mn, le=mx))
    return create_model("PatientData", **fields)

PatientData = build_patient_model(FEATURES or list(ALL_FIELD_META.keys()))


# ── Home page ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    import json
    try:
        # Build only selected feature inputs
        active_features = FEATURES or list(ALL_FIELD_META.keys())
        field_names_js  = json.dumps(active_features)

        fields_html = ""
        for name in active_features:
            mn, mx, val, step = ALL_FIELD_META[name]
            fields_html += (
                f'<label>{name}'
                f'<input id="field_{name}" name="{name}" type="number"'
                f' min="{mn}" max="{mx}" step="{step}" value="{val}" required></label>'
            )

        dropped_html = ""
        if DROPPED:
            dropped_html = (
                '<p class="dropped">Removed low-importance features: '
                + ", ".join(f"<code>{d}</code>" for d in DROPPED)
                + "</p>"
            )

        page = (
            '<!DOCTYPE html><html><head>'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Heart Health ML Demo</title>'
            '<style>'
            'body{font-family:Arial,sans-serif;background:#f3f6fb;margin:0;padding:24px;color:#172033}'
            'main{max-width:850px;margin:auto;background:white;padding:28px;border-radius:16px;box-shadow:0 5px 24px #17203312}'
            'h1{color:#174ea6}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}'
            'label{font-size:14px;font-weight:bold;display:block}'
            'input{display:block;box-sizing:border-box;width:100%;margin-top:7px;padding:11px;border:1px solid #cbd5e1;border-radius:8px;background:white}'
            'button{margin-top:22px;width:100%;padding:14px;border:0;border-radius:8px;background:#174ea6;color:white;font-size:16px;cursor:pointer}'
            '.note{color:#596579;font-size:13px}'
            '.dropped{font-size:13px;color:#7c3aed;background:#f5f3ff;border:1px solid #ddd6fe;border-radius:8px;padding:10px 14px;margin-top:12px}'
            '.dropped code{background:#ede9fe;padding:2px 6px;border-radius:4px;font-size:12px}'
            '#overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.45);z-index:100}'
            '#overlay.show{display:flex;align-items:center;justify-content:center}'
            '#mbox{background:white;border-radius:16px;padding:32px 28px 24px;max-width:420px;width:90%;box-shadow:0 8px 40px rgba(0,0,0,0.18);text-align:center}'
            '#micon{font-size:48px;margin-bottom:12px}'
            '#mtitle{font-size:20px;font-weight:bold;margin-bottom:12px}'
            '#mbody{font-size:15px;line-height:1.7;margin-bottom:24px}'
            '#mclose{padding:10px 36px;border:0;border-radius:8px;font-size:15px;cursor:pointer;color:white}'
            '</style></head><body><main>'
            '<h1>Heart Health Prediction</h1>'
            '<p>Using <strong>' + str(len(active_features)) + ' selected features</strong> (feature selection applied).</p>'
            + dropped_html +
            '<form id="pf" onsubmit="return false"><div class="grid">' + fields_html + '</div>'
            '<button type="submit">Run Prediction</button></form>'
            '<p class="note">Educational ML demo only. Not a medical diagnosis.</p>'
            '</main>'
            '<div id="overlay"><div id="mbox">'
            '<div id="micon"></div><div id="mtitle"></div><div id="mbody"></div>'
            '<button id="mclose" onclick="closeModal()">Close</button>'
            '</div></div>'
            '<script>'
            'var fn=' + field_names_js + ';'
            'function showModal(ok,title,body){'
            '  document.getElementById("micon").textContent=ok?"✓":"✗";'
            '  var t=document.getElementById("mtitle");t.textContent=title;t.style.color=ok?"#16a34a":"#dc2626";'
            '  document.getElementById("mbody").innerHTML=body;'
            '  document.getElementById("mclose").style.background=ok?"#16a34a":"#dc2626";'
            '  document.getElementById("overlay").className="show";'
            '}'
            'function closeModal(){document.getElementById("overlay").className="";}'
            'document.getElementById("overlay").addEventListener("click",function(e){if(e.target===this)closeModal();});'
            'document.getElementById("pf").addEventListener("submit",function(e){'
            '  e.preventDefault();'
            '  var btn=e.target.querySelector("button[type=submit]");'
            '  btn.disabled=true;btn.textContent="Predicting...";'
            '  var d={};'
            '  for(var i=0;i<fn.length;i++){'
            '    var inp=document.getElementById("field_"+fn[i]);'
            '    d[fn[i]]=inp?Number(inp.value):0;'
            '  }'
            '  fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
            '  .then(function(r){var s=r.status;return r.text().then(function(t){return{s:s,t:t};})})'
            '  .then(function(r){'
            '    btn.disabled=false;btn.textContent="Run Prediction";'
            '    var o;try{o=JSON.parse(r.t);}catch(ex){showModal(false,"Server Error",r.t.substring(0,200));return;}'
            '    if(r.s!==200){showModal(false,"Error "+r.s,o.detail||"Prediction failed");return;}'
            '    var lbl=o.prediction===1?"Heart Disease Detected":"No Heart Disease Detected";'
            '    showModal(true,"Prediction Result","<strong>"+lbl+"</strong><br><br>No Heart Disease: <strong>"+(o.probability_class_0*100).toFixed(2)+"%</strong><br>Heart Disease: <strong>"+(o.probability_class_1*100).toFixed(2)+"%</strong>");'
            '  })'
            '  .catch(function(err){btn.disabled=false;btn.textContent="Run Prediction";showModal(false,"Network Error",err.message);});'
            '});'
            '</script></body></html>'
        )
        return Response(content=page, media_type="text/html; charset=utf-8", status_code=200)

    except Exception as ex:
        import traceback
        err = traceback.format_exc()
        err_page = (
            '<!DOCTYPE html><html><head><title>App Error</title>'
            '<style>body{font-family:Arial;padding:32px}pre{background:#fef2f2;padding:16px;border-radius:8px;white-space:pre-wrap}</style>'
            '</head><body><h1 style="color:#dc2626">Server Error</h1>'
            '<pre>' + err.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;') + '</pre>'
            '</body></html>'
        )
        return Response(content=err_page, media_type="text/html; charset=utf-8", status_code=500)


# ── Health & Debug ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "selected_features": FEATURES,
        "dropped_features": DROPPED,
    }


@app.get("/debug")
def debug():
    import sys
    cwd      = os.getcwd()
    file_dir = str(Path(__file__).resolve().parent)
    return {
        "cwd": cwd,
        "file_dir": file_dir,
        "files_in_cwd": os.listdir(cwd),
        "model_loaded": model is not None,
        "selected_features": FEATURES,
        "dropped_features": DROPPED,
        "python": sys.version,
    }


# ── Predict ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    values   = patient.model_dump()
    input_df = pd.DataFrame([values], columns=FEATURES)

    prediction    = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes       = list(model.classes_)

    return {
        "prediction":          prediction,
        "probability_class_0": float(probabilities[classes.index(0)]),
        "probability_class_1": float(probabilities[classes.index(1)]),
        "features_used":       FEATURES,
    }
