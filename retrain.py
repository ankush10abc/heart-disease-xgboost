import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('datasets/Heart_attack.csv').dropna()
ALL_FEATURES = [c for c in df.columns if c != 'HeartDiseaseorAttack']
X = df[ALL_FEATURES]
y = df['HeartDiseaseorAttack']

# ── 2. Feature Selection (3 methods combined) ─────────────────────────────────
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

# Chi-Square
chi2_scores = SelectKBest(chi2, k='all').fit(X, y).scores_

# ANOVA F-Score
f_scores = SelectKBest(f_classif, k='all').fit(X, y).scores_

# Random Forest Importance
rf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fs.fit(X, y)
rf_scores = rf_fs.feature_importances_

# Combined normalized score
combined = pd.DataFrame({
    'Feature':   ALL_FEATURES,
    'chi2':      normalize(chi2_scores),
    'f_score':   normalize(f_scores),
    'rf':        normalize(rf_scores),
})
combined['score'] = (combined['chi2'] + combined['f_score'] + combined['rf']) / 3
combined = combined.sort_values('score', ascending=False)

print("\nFeature Scores:")
print(combined[['Feature', 'score']].to_string(index=False))

# ── 3. Drop low-importance features ──────────────────────────────────────────
THRESHOLD = 0.1
FORCE_DROP = ['Income', 'Education', 'PhysActivity', 'Sex', 'HvyAlcoholConsump', 'CholCheck']
FORCE_KEEP = ['Smoker']

SELECTED = [
    f for f in combined['Feature'].tolist()
    if (combined.loc[combined['Feature'] == f, 'score'].values[0] >= THRESHOLD or f in FORCE_KEEP)
    and f not in FORCE_DROP
]
DROPPED = [f for f in ALL_FEATURES if f not in SELECTED]

print(f"\nThreshold : {THRESHOLD}")
print(f"Kept  ({len(SELECTED)}): {SELECTED}")
print(f"Dropped ({len(DROPPED)}): {DROPPED}")

X_selected = X[SELECTED]

# ── 4. Train model on selected features only ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f'\nAccuracy : {acc:.4f}')

# ── 5. Save bundle ────────────────────────────────────────────────────────────
bundle = {
    'model':    model,
    'features': SELECTED,       # only selected features
    'dropped':  DROPPED,
    'threshold': THRESHOLD,
}
with open('model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

size_mb = os.path.getsize('model.pkl') / 1024 / 1024
print(f'Size     : {size_mb:.2f} MB')
print(f'Saved    : model.pkl  (features={SELECTED})')
