import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------- STEP 1: Generate Synthetic-Realistic Dataset ---------

# Sample data for Cross River
locations = [
    "Abi", "Akamkpa", "Akpabuyo", "Bakassi", "Bekwarra", "Biase", "Boki",
    "Calabar Municipal", "Calabar South", "Etung", "Ikom", "Obanliku",
    "Obubra", "Obudu", "Odukpani", "Ogoja", "Yakurr", "Yala"
]

data = []

for loc in locations:
    for _ in range(200):  # 200 samples per location
        rainfall = np.random.uniform(0, 100, 7).round(1).tolist()
        soil_moisture = round(np.random.uniform(10, 60), 2)
        elevation = round(np.random.uniform(5, 250), 1)

        flood = 1 if sum(rainfall) > 300 or rainfall[-1] > 70 else 0

        data.append(rainfall + [soil_moisture, elevation, flood, loc])

columns = [f'day_{i}' for i in range(1, 8)] + ['soil_moisture', 'elevation', 'flood', 'location']
df = pd.DataFrame(data, columns=columns)

# --------- STEP 2: Train Model ---------

features = [f'day_{i}' for i in range(1, 8)] + ['soil_moisture', 'elevation']
X = df[features]
y = df['flood']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'flood_model.pkl')
print("✅ Model trained and saved as flood_model.pkl")
