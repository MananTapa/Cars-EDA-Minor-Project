"""
Exploratory Data Analysis on a Cars Dataset
Student: Manan Thapa

The project uses the public 'CAR DETAILS FROM CAR DEKHO' dataset.
It focuses only on EDA; no machine-learning model is used.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/bagassenop/cardekho/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
OUTPUT_DIR = Path("cars_eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# 1. Load and inspect the data
cars = pd.read_csv(DATA_URL)
print("\n--- DATASET OVERVIEW ---")
print("Shape:", cars.shape)
print("\nColumns:", list(cars.columns))
print("\nFirst five rows:\n", cars.head())
print("\nData types:\n", cars.dtypes)

numeric_cols = cars.select_dtypes(include="number").columns.tolist()
categorical_cols = cars.select_dtypes(exclude="number").columns.tolist()
print("\nNumerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# 2. Data-quality checks
rows_before = len(cars)
print("\n--- DATA QUALITY ---")
print("Missing values:\n", cars.isna().sum())
print("Duplicate rows:", cars.duplicated().sum())

# Standardise text fields before checking categories.
for col in categorical_cols:
    cars[col] = cars[col].astype(str).str.strip()

# Exact duplicate listings do not add new information to this EDA.
cars = cars.drop_duplicates().copy()
rows_after = len(cars)
print(f"Rows before cleaning: {rows_before}")
print(f"Rows after removing exact duplicates: {rows_after}")

# Brand is the first word in the car name. Age is measured from the latest year in this dataset.
cars["brand"] = cars["name"].str.split().str[0]
reference_year = int(cars["year"].max())
cars["car_age"] = reference_year - cars["year"]

# IQR flags are used to inspect outliers, not to automatically delete them.
def iqr_outlier_count(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum()), lower, upper

for col in ["selling_price", "km_driven"]:
    count, lower, upper = iqr_outlier_count(cars[col])
    print(f"{col}: {count} IQR outliers (limits {lower:,.0f} to {upper:,.0f})")

# 3. Descriptive statistics
print("\n--- NUMERICAL SUMMARY ---")
print(cars[["year", "selling_price", "km_driven", "car_age"]].describe().round(2))

print("\n--- CATEGORICAL SUMMARY ---")
for col in ["fuel", "seller_type", "transmission", "owner", "brand"]:
    print(f"\n{col}:\n{cars[col].value_counts().head(10)}")

# Helper used for all saved figures.
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()

# 4. Univariate analysis
plt.figure(figsize=(8, 5))
sns.histplot(cars["selling_price"] / 100000, bins=35, kde=True)
plt.title("Distribution of Used-Car Selling Prices")
plt.xlabel("Selling price (lakh INR)")
plt.ylabel("Number of cars")
save_plot("01_price_distribution.png")
print("Observation 1: Prices are right-skewed; most listings are in the lower and middle price range, with a small premium-car tail.")

plt.figure(figsize=(8, 5))
sns.countplot(data=cars, x="fuel", order=cars["fuel"].value_counts().index)
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel type")
plt.ylabel("Number of cars")
plt.xticks(rotation=20)
save_plot("02_fuel_type.png")
print("Observation 2: Diesel and petrol dominate the listings; alternative fuel types form a small share.")

# 5. Bivariate analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(data=cars, x="car_age", y=cars["selling_price"] / 100000, alpha=0.45)
plt.title("Vehicle Age and Selling Price")
plt.xlabel(f"Car age in years (relative to {reference_year})")
plt.ylabel("Selling price (lakh INR)")
save_plot("03_age_vs_price.png")
print("Observation 3: Newer cars generally command higher prices, although brand and segment create considerable variation.")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=cars, x="km_driven", y=cars["selling_price"] / 100000, alpha=0.4)
plt.title("Kilometres Driven and Selling Price")
plt.xlabel("Kilometres driven")
plt.ylabel("Selling price (lakh INR)")
plt.xlim(0, cars["km_driven"].quantile(0.99))
save_plot("04_km_vs_price.png")
print("Observation 4: Higher usage tends to be associated with lower prices, but mileage alone does not explain price.")

plt.figure(figsize=(8, 5))
sns.boxplot(data=cars, x="transmission", y=cars["selling_price"] / 100000, showfliers=False)
plt.title("Selling Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Selling price (lakh INR)")
save_plot("05_transmission_price.png")
print("Observation 5: Automatic cars have a higher typical price in this dataset, partly because many belong to premium segments.")

# 6. Brand comparison
brand_summary = (cars.groupby("brand")
                 .agg(listings=("name", "size"), average_price=("selling_price", "mean"))
                 .query("listings >= 20")
                 .sort_values("average_price", ascending=False)
                 .head(10)
                 .sort_values("average_price"))

plt.figure(figsize=(8, 5.5))
sns.barplot(x=brand_summary["average_price"] / 100000, y=brand_summary.index)
plt.title("Highest Average Price Brands (minimum 20 listings)")
plt.xlabel("Average selling price (lakh INR)")
plt.ylabel("Brand")
save_plot("06_brand_average_price.png")
print("Observation 6: Premium brands occupy the upper end of average resale prices; high-volume mass-market brands are generally lower.")

# 7. Ownership and price
owner_order = cars.groupby("owner")["selling_price"].median().sort_values(ascending=False).index
plt.figure(figsize=(9, 5))
sns.boxplot(data=cars, x="owner", y=cars["selling_price"] / 100000, order=owner_order, showfliers=False)
plt.title("Selling Price by Ownership History")
plt.xlabel("Ownership category")
plt.ylabel("Selling price (lakh INR)")
plt.xticks(rotation=25, ha="right")
save_plot("07_owner_price.png")
print("Observation 7: First-owner cars generally retain better prices than cars with multiple previous owners.")

# 8. Multivariate correlation for numeric variables
corr = cars[["year", "selling_price", "km_driven", "car_age"]].corr()
plt.figure(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Numerical Variables")
save_plot("08_correlation_heatmap.png")
print("Observation 8: Year/age shows a clearer relationship with price than kilometres driven; age and year are naturally inverse.")

# A small summary file is useful while checking the project output.
summary = pd.DataFrame({
    "metric": ["raw_rows", "clean_rows", "duplicates_removed", "average_price", "median_price", "average_km_driven"],
    "value": [rows_before, rows_after, rows_before - rows_after,
              round(cars["selling_price"].mean(), 2),
              round(cars["selling_price"].median(), 2),
              round(cars["km_driven"].mean(), 2)]
})
summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)

print("\n--- FINAL FINDINGS ---")
print("1. Used-car prices are strongly right-skewed because a small number of premium cars are much more expensive.")
print("2. Petrol and diesel cars make up almost the entire dataset.")
print("3. Vehicle age has a clear negative relationship with selling price.")
print("4. Higher kilometres driven generally reduces resale value, although the relationship is noisy.")
print("5. Automatic cars show higher typical prices than manual cars in this sample.")
print("6. Brand and ownership history also create visible differences in resale price.")
print(f"\nCharts and the summary file were saved in: {OUTPUT_DIR.resolve()}")
