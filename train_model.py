
from pathlib import Path
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading California Housing dataset...")

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 4))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

model_dir = Path(__file__).resolve().parent / "model"
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "model.pkl")
print("Model saved to model/model.pkl")
  