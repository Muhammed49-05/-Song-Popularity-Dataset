import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\Users\Eng Gamal\Documents\SongPopularity.csv")

# Data cleaning
df = df.drop_duplicates() # Remove duplicates
df = df.dropna() # Remove rows with missing values

# Convert categorical columns to numerical
categorical_cols = ['Artist Names', 'Artist(s) Genres', 'Song Image']
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Select numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation analysis
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Mutual information
mutual_info = mutual_info_regression(numeric_df, df['Popularity'])
top_features = numeric_df.columns[mutual_info.argsort()[-10:]]
print(top_features)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(numeric_df[top_features], df['Popularity'])
rf_pred = rf_model.predict(numeric_df[top_features])

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(numeric_df[top_features], df['Popularity'])
gb_pred = gb_model.predict(numeric_df[top_features])

# Model evaluation
models = {'Random Forest Regressor': rf_model, 'Gradient Boosting Regressor': gb_model}
for model_name, model in models.items():
    pred = model.predict(numeric_df[top_features])
    mse = mean_squared_error(df['Popularity'], pred)
    mae = mean_absolute_error(df['Popularity'], pred)
    r2 = r2_score(df['Popularity'], pred)
    
    print(f"{model_name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")