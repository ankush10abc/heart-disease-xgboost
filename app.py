
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Prediction Demo")

MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(MODEL_PATH, "rb") as file:
    bundle = pickle.load(file)

model = bundle["model"]
FEATURES = bundle["features"]


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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Heart Health ML Demo</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background: #f3f6fb;
          margin: 0;
          padding: 24px;
          color: #172033;
        }
        main {
          max-width: 850px;
          margin: auto;
          background: white;
          padding: 28px;
          border-radius: 16px;
          box-shadow: 0 5px 24px #17203312;
        }
        h1 { color: #174ea6; }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 16px;
        }
        label { font-size: 14px; font-weight: bold; }
        input, select {
          display: block;
          box-sizing: border-box;
          width: 100%;
          margin-top: 7px;
          padding: 11px;
          border: 1px solid #cbd5e1;
          border-radius: 8px;
          background: white;
        }
        button {
          margin-top: 22px;
          width: 100%;
          padding: 14px;
          border: 0;
          border-radius: 8px;
          background: #174ea6;
          color: white;
          font-size: 16px;
          cursor: pointer;
        }
        #result {
          margin-top: 20px;
          padding: 16px;
          border-radius: 8px;
          background: #eef4ff;
          white-space: pre-line;
        }
        .note { color: #596579; font-size: 13px; }
      </style>
    </head>
    <body>
    <main>
      <h1>Heart Health Prediction</h1>
      <p>Enter the coded features used by the training dataset.</p>

      <form id="patientForm">
        <div class="grid" id="fields"></div>
        <button type="submit">Run Prediction</button>
      </form>

      <div id="result" hidden></div>
      <p class="note">
        Educational ML demo only. This tool is not a medical diagnosis.
      </p>
    </main>

    <script>
      const fields = [
        ["HighBP", 0, 1, 1],
        ["HighChol", 0, 1, 1],
        ["CholCheck", 0, 1, 1],
        ["BMI", 10, 70, 30],
        ["Smoker", 0, 1, 0],
        ["Stroke", 0, 1, 0],
        ["Diabetes", 0, 2, 0],
        ["PhysActivity", 0, 1, 1],
        ["HvyAlcoholConsump", 0, 1, 0],
        ["MentHlth", 0, 30, 0],
        ["PhysHlth", 0, 30, 0],
        ["Sex", 0, 1, 0],
        ["Age", 1, 13, 7],
        ["Education", 1, 6, 4],
        ["Income", 1, 8, 5]
      ];

      const container = document.getElementById("fields");

      for (const [name, min, max, initial] of fields) {
        const label = document.createElement("label");
        label.textContent = name;

        const input = document.createElement("input");
        input.name = name;
        input.type = "number";
        input.min = min;
        input.max = max;
        input.step = name === "BMI" ? "0.1" : "1";
        input.value = initial;
        input.required = true;

        label.appendChild(input);
        container.appendChild(label);
      }

      document.getElementById("patientForm")
        .addEventListener("submit", async (event) => {
          event.preventDefault();

          const result = document.getElementById("result");
          result.hidden = false;
          result.textContent = "Predicting...";

          const data = {};
          for (const [name] of fields) {
            data[name] = Number(
              document.querySelector(`[name="${name}"]`).value
            );
          }

          try {
            const response = await fetch("/predict", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify(data)
            });

            const output = await response.json();

            if (!response.ok) {
              throw new Error(output.detail || "Prediction failed");
            }

            result.textContent =
              "Predicted class: " + output.prediction +
              "\\nEstimated probability of class 0: " +
              (output.probability_class_0 * 100).toFixed(2) + "%" +
              "\\nEstimated probability of class 1: " +
              (output.probability_class_1 * 100).toFixed(2) + "%";
          } catch (error) {
            result.textContent = "Error: " + error.message;
          }
        });
    </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(patient: PatientData):
    values = patient.model_dump()

    # Keep exactly the feature order used during training.
    input_df = pd.DataFrame([values], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    probability_0 = float(probabilities[classes.index(0)])
    probability_1 = float(probabilities[classes.index(1)])

    return {
        "prediction": prediction,
        "probability_class_0": probability_0,
        "probability_class_1": probability_1
    }
