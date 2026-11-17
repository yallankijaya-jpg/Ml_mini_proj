============================================

WALMART SALES PREDICTION

============================================

1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

============================================

2. Load dataset

============================================

data = pd.read_csv(r"E:\y24ads531\traindata for ml project.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

============================================

3. Basic information

============================================

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

============================================

4. Convert Date column

============================================

data["Date"] = pd.to_datetime(data["Date"])

============================================

5. Create simple date features

============================================

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Week"] = data["Date"].dt.isocalendar().week.astype(int)

============================================

6. Exploratory Data Analysis

============================================

Total sales by year

year_sales = data.groupby("Year")["Weekly_Sales"].sum()

plt.figure(figsize=(8, 5))
year_sales.plot(kind="bar")

plt.title("Total Sales by Year")
plt.xlabel("Year")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

--------------------------------------------

Sales by store

--------------------------------------------

store_sales = data.groupby("Store")["Weekly_Sales"].sum()

plt.figure(figsize=(10, 5))

store_sales.plot(kind="bar")

plt.title("Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

--------------------------------------------

Monthly sales

--------------------------------------------

month_sales = data.groupby("Month")["Weekly_Sales"].mean()

plt.figure(figsize=(8, 5))

month_sales.plot(marker="o")

plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Weekly Sales")

plt.grid()
plt.tight_layout()
plt.show()

--------------------------------------------

Sales distribution

--------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
data["Weekly_Sales"],
bins=40,
kde=True
)

plt.title("Distribution of Weekly Sales")
plt.xlabel("Weekly Sales")

plt.tight_layout()
plt.show()

============================================

7. Select features

============================================

X = data[
[
"Store",
"Dept",
"IsHoliday",
"Year",
"Month",
"Week"
]
]

y = data["Weekly_Sales"]

============================================

8. Convert Boolean to integer

============================================

X["IsHoliday"] = X["IsHoliday"].astype(int)

============================================

9. Split dataset

============================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

============================================

10. Linear Regression

============================================

linear_model = LinearRegression()

linear_model.fit(
X_train,
y_train
)

linear_prediction = linear_model.predict(
X_test
)

============================================

11. Evaluate Linear Regression

============================================

linear_mae = mean_absolute_error(
y_test,
linear_prediction
)

linear_r2 = r2_score(
y_test,
linear_prediction
)

print("\nLinear Regression Results")
print("----------------------------")
print("MAE:", linear_mae)
print("R2 Score:", linear_r2)

============================================

12. Random Forest

============================================

random_forest = RandomForestRegressor(
n_estimators=100,
random_state=42,
n_jobs=-1
)

random_forest.fit(
X_train,
y_train
)

rf_prediction = random_forest.predict(
X_test
)

============================================

13. Evaluate Random Forest

============================================

rf_mae = mean_absolute_error(
y_test,
rf_prediction
)

rf_r2 = r2_score(
y_test,
rf_prediction
)

print("\nRandom Forest Results")
print("----------------------------")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)

============================================

14. Compare models

============================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
"Linear Regression R2:",
linear_r2
)

print(
"Random Forest R2:",
rf_r2
)

============================================

15. Actual vs Predicted Sales

============================================

plt.figure(figsize=(10, 5))

plt.scatter(
y_test,
rf_prediction,
alpha=0.3
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
"Actual vs Predicted Walmart Sales"
)

plt.tight_layout()
plt.show()

============================================

16. Feature Importance

============================================

importance = pd.Series(
random_forest.feature_importances_,
index=X.columns
)

importance = importance.sort_values(
ascending=False
)

plt.figure(figsize=(8, 5))

importance.plot(
kind="bar"
)

plt.title(
"Feature Importance"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

============================================

17. Make a sample prediction

============================================

sample = pd.DataFrame({
"Store": [1],
"Dept": [1],
"IsHoliday": [0],
"Year": [2012],
"Month": [12],
"Week": [50]
})

prediction = random_forest.predict(
sample
)

print("\n================================")
print("SAMPLE SALES PREDICTION")
print("================================")

print(
"Predicted Weekly Sales:",
round(prediction[0], 2)
)

============================================

18. Finish

============================================

print("\nProject completed successfully!")     explain it in deail step by step
