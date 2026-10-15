
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data = fetch_california_housing(as_frame=True)

X = data.data.iloc[:, :10] if data.data.shape[1] >= 10 else data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model: Random Forest Regressor")
print("Number of Features:", X.shape[1])
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

new_house = pd.DataFrame(
    [X.median().to_dict()],
    columns=X.columns
)

predicted_price = model.predict(new_house)

print("Predicted House Value:", predicted_price[0])