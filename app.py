import pickle
import os
import json
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
    for path in [
        Path(__file__).resolve().parent / "model.pkl",
        Path(os.getcwd()) / "model.pkl",
        Path("/var/task/model.pkl"),
    ]:
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            model    = bundle["model"]
            FEATURES = bundle["features"]
            DROPPED  = bundle.get("dropped", ["CholCheck", "PhysActivity", "HvyAlcoholConsump", "Sex", "Education", "Income"])
            print(f"Model loaded: {path} | features={FEATURES}")
            return
        except Exception as e:
            print(f"Failed {path}: {e}")
    print("ERROR: model.pkl not found")

load_model()

# ── Field UI config: label, hint, min, max, default, step ────────────────────
FIELD_UI = {
    "PhysHlth": ("Physical Health Days", "Days of poor physical health (0–30)",  0,  30, 0,   "1"),
    "Age":      ("Age Group",            "1=18-24, 7=50-54, 13=80+",             1,  13, 7,   "1"),
    "HighBP":   ("High Blood Pressure",  "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "Stroke":   ("Stroke History",       "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "BMI":      ("BMI",                  "Body Mass Index (10.0 – 70.0)",         10, 70, 27,  "0.1"),
    "Diabetes": ("Diabetes",             "0 = No  |  1 = Pre-diabetic  |  2 = Yes", 0, 2, 0, "1"),
    "HighChol": ("High Cholesterol",     "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "MentHlth": ("Mental Health Days",   "Days of poor mental health (0–30)",     0,  30, 0,   "1"),
    "Smoker":   ("Smoker",               "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    # kept for fallback if model has extra features
    "HighBP":   ("High Blood Pressure",  "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "CholCheck":("Cholesterol Check",    "0 = No  |  1 = Yes",                   0,  1,  1,   "1"),
    "Smoker":   ("Smoker",               "0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "PhysActivity":("Physical Activity", "0 = No  |  1 = Yes",                   0,  1,  1,   "1"),
    "HvyAlcoholConsump":("Heavy Alcohol","0 = No  |  1 = Yes",                   0,  1,  0,   "1"),
    "Sex":      ("Sex",                  "0 = Female  |  1 = Male",              0,  1,  0,   "1"),
    "Education":("Education Level",      "Scale 1–6",                            1,  6,  4,   "1"),
    "Income":   ("Income Level",         "Scale 1–8",                            1,  8,  5,   "1"),
}

# ── Build Pydantic model dynamically from selected features only ──────────────
def build_patient_model(features):
    fields = {}
    for name in features:
        _, _, mn, mx, default, _ = FIELD_UI[name]
        if name == "BMI":
            fields[name] = (float, Field(default=float(default), gt=0, le=float(mx)))
        else:
            fields[name] = (int, Field(default=int(default), ge=int(mn), le=int(mx)))
    return create_model("PatientData", **fields)

PatientData = build_patient_model(FEATURES or list(FIELD_UI.keys()))


# ── Home page ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        active  = FEATURES or list(FIELD_UI.keys())
        dropped = DROPPED  or []

        # ── Verify no dropped field leaks into UI ──
        assert not any(f in set(dropped) for f in active), "Dropped feature leaked into UI"

        # ── Build input cards ──
        cards = ""
        for name in active:
            label, hint, mn, mx, val, step = FIELD_UI[name]
            cards += (
                f'<div class="card">'
                f'<label for="f_{name}">{label}</label>'
                f'<span class="hint">{hint}</span>'
                f'<input id="f_{name}" name="{name}" type="number"'
                f' min="{mn}" max="{mx}" step="{step}" value="{val}" required'
                f' oninput="liveValidate(this,{mn},{mx})">'
                f'<span class="err" id="e_{name}"></span>'
                f'</div>'
            )

        fields_js = json.dumps(active)

        page = (
            '<!DOCTYPE html>'
            '<html lang="en"><head>'
            '<meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Heart Disease Risk Predictor</title>'
            '<style>'
            '*{box-sizing:border-box;margin:0;padding:0}'
            'body{font-family:"Segoe UI",Arial,sans-serif;background:linear-gradient(135deg,#e8f0fe 0%,#fce4ec 100%);min-height:100vh;padding:32px 16px}'
            '.wrap{max-width:920px;margin:auto}'

            # header
            '.hdr{text-align:center;margin-bottom:28px}'
            '.hdr h1{font-size:2rem;color:#1a237e;font-weight:700}'
            '.hdr p{color:#546e7a;margin-top:6px;font-size:.95rem}'
            '.pill{display:inline-flex;align-items:center;gap:6px;background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;border-radius:20px;padding:5px 16px;font-size:13px;margin-top:10px;font-weight:600}'

            # dropped bar
            '.drop-bar{display:none}'

            # card grid
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin-bottom:26px}'
            '.card{background:white;border-radius:12px;padding:18px 16px;box-shadow:0 2px 10px rgba(0,0,0,.07);border:1.5px solid #e3e8f0;transition:box-shadow .2s,border-color .2s}'
            '.card:focus-within{border-color:#3f51b5;box-shadow:0 4px 18px rgba(63,81,181,.15)}'
            '.card label{display:block;font-size:12px;font-weight:700;color:#1a237e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}'
            '.hint{display:block;font-size:11px;color:#90a4ae;margin-bottom:8px}'
            '.card input{width:100%;padding:10px 12px;border:1.5px solid #e3e8f0;border-radius:8px;font-size:15px;color:#172033;outline:none;transition:border .2s}'
            '.card input:focus{border-color:#3f51b5}'
            '.card input.bad{border-color:#e53935;background:#fff5f5}'
            '.err{display:block;font-size:11px;color:#e53935;margin-top:4px;min-height:15px}'

            # submit
            '.btn-row{text-align:center;margin-bottom:8px}'
            '#submitBtn{background:linear-gradient(90deg,#1a237e,#3f51b5);color:white;border:0;border-radius:10px;padding:15px 64px;font-size:16px;font-weight:600;cursor:pointer;box-shadow:0 4px 16px rgba(63,81,181,.35);transition:opacity .2s,transform .1s;letter-spacing:.3px}'
            '#submitBtn:hover{opacity:.91;transform:translateY(-1px)}'
            '#submitBtn:disabled{opacity:.55;cursor:not-allowed;transform:none}'
            '.note{text-align:center;color:#90a4ae;font-size:12px;margin-top:14px}'

            # modal
            '#ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.52);z-index:300;align-items:center;justify-content:center}'
            '#ov.show{display:flex}'
            '#mb{background:white;border-radius:20px;padding:36px 30px 28px;max-width:450px;width:93%;box-shadow:0 12px 48px rgba(0,0,0,.22);text-align:center;animation:pop .2s ease}'
            '@keyframes pop{from{transform:scale(.85);opacity:0}to{transform:scale(1);opacity:1}}'
            '#mi{font-size:54px;margin-bottom:10px}'
            '#mt{font-size:21px;font-weight:700;margin-bottom:8px}'
            '#md{font-size:14px;line-height:1.8;color:#455a64;margin-bottom:22px}'
            '.bar-wrap{margin:12px 0 2px;text-align:left}'
            '.bar-lbl{display:flex;justify-content:space-between;font-size:12px;color:#78909c;margin-bottom:4px}'
            '.bar-bg{height:11px;border-radius:6px;background:#e3e8f0;overflow:hidden}'
            '.bar-fill{height:100%;border-radius:6px;transition:width .7s ease}'
            '#mc{padding:11px 44px;border:0;border-radius:10px;font-size:15px;cursor:pointer;color:white;font-weight:600}'
            '</style></head><body>'
            '<div class="wrap">'

            # header
            '<div class="hdr">'
            '<h1>&#10084;&#65039; Heart Disease Risk Predictor</h1>'
            '<p>AI-powered prediction using clinical health indicators</p>'
            '<span class="pill">&#10003;&nbsp;' + str(len(active)) + ' Features&nbsp;&nbsp;|&nbsp;&nbsp;Random Forest&nbsp;&nbsp;|&nbsp;&nbsp;90.8% Accuracy</span>'
            '</div>'

            # dropped bar (hidden - field names must not appear in page source)
            '<div class="drop-bar" style="display:none"></div>'

            # form
            '<form id="pf" novalidate>'
            '<div class="grid">' + cards + '</div>'
            '<div class="btn-row"><button type="submit" id="submitBtn">&#128269;&nbsp;Run Prediction</button></div>'
            '</form>'
            '<p class="note">&#9888;&#65039; ML demo only &mdash; not a substitute for medical advice.</p>'
            '</div>'

            # modal
            '<div id="ov"><div id="mb">'
            '<div id="mi"></div><div id="mt"></div><div id="md"></div>'
            '<button id="mc">Close</button>'
            '</div></div>'

            '<script>'
            'var F=' + fields_js + ';'

            'function liveValidate(inp,mn,mx){'
            '  var v=Number(inp.value),e=document.getElementById("e_"+inp.name);'
            '  if(inp.value===""||isNaN(v)||v<mn||v>mx){'
            '    inp.classList.add("bad");'
            '    e.textContent="Must be "+mn+" – "+mx;return false;'
            '  }'
            '  inp.classList.remove("bad");e.textContent="";return true;'
            '}'

            'function showModal(ok,title,p0,p1){'
            '  document.getElementById("mi").textContent=ok?"\\u2764\\uFE0F":"\\u26A0\\uFE0F";'
            '  var t=document.getElementById("mt");'
            '  t.textContent=title;'
            '  t.style.color=ok?(p1>50?"#c62828":"#2e7d32"):"#e65100";'
            '  var body="";'
            '  if(p0!==null){'
            '    body+="<div class=\\"bar-wrap\\"><div class=\\"bar-lbl\\"><span>No Disease</span><span>"+p0.toFixed(1)+"%</span></div>"'
            '         +"<div class=\\"bar-bg\\"><div class=\\"bar-fill\\" style=\\"width:"+p0+"%;background:#43a047\\"></div></div></div>";'
            '    body+="<div class=\\"bar-wrap\\"><div class=\\"bar-lbl\\"><span>Heart Disease</span><span>"+p1.toFixed(1)+"%</span></div>"'
            '         +"<div class=\\"bar-bg\\"><div class=\\"bar-fill\\" style=\\"width:"+p1+"%;background:#e53935\\"></div></div></div>";'
            '  } else { body=title; document.getElementById("mt").textContent="Error"; }'
            '  document.getElementById("md").innerHTML=body;'
            '  var mc=document.getElementById("mc");'
            '  mc.style.background=ok?(p1>50?"#c62828":"#2e7d32"):"#e65100";'
            '  document.getElementById("ov").className="show";'
            '}'

            'document.getElementById("mc").onclick=function(){document.getElementById("ov").className="";};'
            'document.getElementById("ov").addEventListener("click",function(e){if(e.target===this)document.getElementById("ov").className="";});'

            'document.getElementById("pf").addEventListener("submit",function(e){'
            '  e.preventDefault();'
            '  var ok=true;'
            '  for(var i=0;i<F.length;i++){'
            '    var inp=document.getElementById("f_"+F[i]);'
            '    if(inp&&!liveValidate(inp,Number(inp.min),Number(inp.max)))ok=false;'
            '  }'
            '  if(!ok)return;'
            '  var btn=document.getElementById("submitBtn");'
            '  btn.disabled=true;btn.textContent="Predicting...";'
            '  var d={};'
            '  for(var j=0;j<F.length;j++){'
            '    var el=document.getElementById("f_"+F[j]);'
            '    d[F[j]]=el?Number(el.value):0;'
            '  }'
            '  fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
            '  .then(function(r){var s=r.status;return r.text().then(function(t){return{s:s,t:t};});})'
            '  .then(function(r){'
            '    btn.disabled=false;btn.innerHTML="&#128269;&nbsp;Run Prediction";'
            '    var o;try{o=JSON.parse(r.t);}catch(ex){showModal(false,"Server Error",null,null);return;}'
            '    if(r.s!==200){showModal(false,o.detail||"Prediction failed",null,null);return;}'
            '    var p0=o.probability_class_0*100,p1=o.probability_class_1*100;'
            '    var lbl=o.prediction===1?"\\u26A0\\uFE0F Heart Disease Risk Detected":"\\u2705 No Heart Disease Detected";'
            '    showModal(true,lbl,p0,p1);'
            '  })'
            '  .catch(function(err){'
            '    btn.disabled=false;btn.innerHTML="&#128269;&nbsp;Run Prediction";'
            '    showModal(false,"Network Error: "+err.message,null,null);'
            '  });'
            '});'
            '</script></body></html>'
        )
        return Response(content=page, media_type="text/html; charset=utf-8", status_code=200)

    except Exception as ex:
        import traceback
        err = traceback.format_exc()
        err_page = (
            '<!DOCTYPE html><html><head><title>App Error</title>'
            '<style>body{font-family:Arial;padding:32px}pre{background:#fef2f2;padding:16px;border-radius:8px;white-space:pre-wrap;color:#7f1d1d}</style>'
            '</head><body><h1 style="color:#dc2626">Server Error</h1>'
            '<pre>' + err.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') + '</pre>'
            '</body></html>'
        )
        return Response(content=err_page, media_type="text/html; charset=utf-8", status_code=500)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "selected_features": FEATURES,
        "dropped_features": DROPPED,
    }


# ── Debug ─────────────────────────────────────────────────────────────────────
@app.get("/debug")
def debug():
    import sys
    cwd = os.getcwd()
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
