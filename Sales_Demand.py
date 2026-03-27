# -----------------------------
# 📦 Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 📂 Load Dataset (or create if not exists)
# -----------------------------
try:
    df = pd.read_csv("sales_data.csv")
except:
    print("Dataset not found. Creating sample dataset...")
    df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=200),
        "Sales": np.random.randint(100, 500, 200)
    })
    df.to_csv("sales_data.csv", index=False)

# -----------------------------
# 🧹 Data Preprocessing
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# -----------------------------
# ⚙️ Feature Engineering
# -----------------------------
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

df['Lag_1'] = df['Sales'].shift(1)
df['Rolling_Mean'] = df['Sales'].rolling(5).mean()   # changed window

df = df.dropna()

# -----------------------------
# 🎯 Prepare Data
# -----------------------------
X = df[['Day', 'Month', 'Year', 'DayOfWeek', 'Lag_1', 'Rolling_Mean']]
y = df['Sales']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 🤖 Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 📊 Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# -----------------------------
# 🔮 Future Forecast
# -----------------------------
future_dates = pd.date_range(start=df['Date'].max(), periods=30)

future_df = pd.DataFrame({'Date': future_dates})
future_df['Day'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Year'] = future_df['Date'].dt.year
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek

future_df['Lag_1'] = df['Sales'].iloc[-1]
future_df['Rolling_Mean'] = df['Rolling_Mean'].iloc[-1]

future_X = future_df[['Day', 'Month', 'Year', 'DayOfWeek', 'Lag_1', 'Rolling_Mean']]
future_df['Predicted_Sales'] = model.predict(future_X)

# =============================
# 📈 VISUALIZATIONS (UPDATED)
# =============================

# 1️⃣ Actual vs Predicted (added markers)
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Sales'], marker='o', label='Actual Sales')
plt.plot(df['Date'][split:], y_pred, linestyle='dashed', marker='x', label='Predicted Sales')
plt.title("Actual vs Predicted Sales (Improved View)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Actual_vs_Predicted_Sales.png")
plt.show()

# 2️⃣ Sales Trend (added rolling mean line)
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Sales'], label="Sales")
plt.plot(df['Date'], df['Rolling_Mean'], linestyle='dashed', label="Rolling Mean")
plt.title("Sales Trend with Smoothing")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.savefig("Sales_Trend.png")
plt.show()

# 3️⃣ Monthly Sales (line instead of bar)
monthly_sales = df.groupby(df['Date'].dt.month)['Sales'].mean()

plt.figure(figsize=(8,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title("Average Monthly Sales (Line View)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.savefig("Sales_Comparison.png")
plt.show()

# 4️⃣ Sales Distribution (added density feel)
plt.figure(figsize=(8,5))
plt.hist(df['Sales'], bins=20, edgecolor='black')
plt.title("Sales Distribution (Enhanced)")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(True, axis='y')
plt.savefig("Sales_Distribution.png")
plt.show()

# 5️⃣ Future Forecast (added markers + style)
plt.figure(figsize=(10,5))
plt.plot(future_df['Date'], future_df['Predicted_Sales'], marker='o', linestyle='dashed')
plt.title("Future Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("Future_Forecast.png")
plt.show()

# -----------------------------
# 📤 Output
# -----------------------------
print("\nFuture Predictions:")
print(future_df.head())