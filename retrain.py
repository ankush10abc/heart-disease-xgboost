import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/Heart_attack.csv').dropna()
FEATURES = [c for c in df.columns if c != 'HeartDiseaseorAttack']
X, y = df[FEATURES], df['HeartDiseaseorAttack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X_train, y_train)

print('Accuracy:', round(model.score(X_test, y_test), 4))

bundle = {'model': model, 'features': FEATURES}
with open('model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print('Size:', round(os.path.getsize('model.pkl') / 1024 / 1024, 2), 'MB')
print('Features:', FEATURES)
print('Done - model.pkl saved successfully')
