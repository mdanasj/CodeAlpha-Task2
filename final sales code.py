# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 2: Load the data
df = pd.read_csv(r"C:\Users\Anas\CODEALPHA\sales_data.csv.csv")

# Step 3: Initial Cleaning
df.dropna(inplace=True)  # Remove missing rows

# Step 4: Encode Categorical Features (platform and segment)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Step 5: Feature Selection
features = [col for col in df.columns if col != 'Sales']
X = df[features]
y = df['Sales']

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Step 10: Analyze Advertising Impact
ad_cols = [col for col in X.columns if 'Advertising_Spend' in col or 'Platform_' in col]
ad_effects = pd.Series(model.coef_, index=X.columns)[ad_cols]
print("\nAdvertising impact (coefficients):")
print(ad_effects.sort_values(ascending=False))

# Step 11: Visualize
coefficients = pd.Series(model.coef_, index=X.columns)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Coefficients from your model
coefficients = pd.Series(model.coef_, index=X.columns)

# Set vibrant style
sns.set_style("whitegrid")
palette = sns.color_palette("husl", len(coefficients))  # Bright, varied hues

plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients.values, y=coefficients.index, palette=palette, edgecolor="black", saturation=1)

plt.title("📈 Impact of Advertising Channels on Sales", fontsize=14, fontweight='bold', color='navy')
plt.xlabel("Coefficient Value", fontsize=12)
plt.ylabel("Channel", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


