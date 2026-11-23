import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/bagassenop/cardekho/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
df = pd.read_csv(url)

# Make column headings easier to read
df.columns = df.columns.str.replace("_", " ", regex=False).str.upper()

# Dataset understanding
print("Dataset shape:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nSample rows:\n", df.head())
print("\nData types:\n", df.dtypes)

numerical_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Data quality checks
rows_before = len(df)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

print("\nCategories before cleaning:")
for col in categorical_cols:
    print(col, ":", df[col].dropna().unique())

# Data cleaning
# Text labels are stripped so accidental spaces do not create separate categories.
for col in categorical_cols:
    df[col] = df[col].str.strip()

# Exact duplicate rows are removed because they repeat the same information.
df = df.drop_duplicates().copy()

# The existing data types are suitable, so no forced type conversion is needed.
# Missing values are checked above and are only treated if a column actually needs it.

rows_after = len(df)
print("\nRows before cleaning:", rows_before)
print("Rows after cleaning:", rows_after)
print("Rows removed as duplicates:", rows_before - rows_after)

print("\nCategories after cleaning:")
for col in ["FUEL", "SELLER TYPE", "TRANSMISSION", "OWNER"]:
    print(col, ":", df[col].dropna().unique())

# Useful fields for analysis
df["BRAND"] = df["NAME"].str.split().str[0]
df["CAR AGE"] = df["YEAR"].max() - df["YEAR"]

# Outlier check using IQR
# Outliers are identified, but genuine high-price or high-mileage cars are not removed automatically.
for col in ["SELLING PRICE", "KM DRIVEN"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outlier_count = ((df[col] < lower_limit) | (df[col] > upper_limit)).sum()
    print(f"\n{col} IQR limits: {lower_limit:.2f} to {upper_limit:.2f}")
    print(f"{col} outliers identified: {outlier_count}")

print("\nOutliers were retained because unusual price or mileage values can represent genuine vehicles.")

# Descriptive statistics
print("\nNumerical summary:\n", df[["YEAR", "SELLING PRICE", "KM DRIVEN", "CAR AGE"]].describe())

for col in ["FUEL", "SELLER TYPE", "TRANSMISSION", "OWNER", "BRAND"]:
    print(f"\n{col} summary:\n", df[col].value_counts().head(10))

# Univariate analysis
plt.figure(figsize=(8, 5))
sns.histplot(df["SELLING PRICE"] / 100000, bins=30, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (Lakh INR)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()
print("Observation: Most cars are in the lower and middle price range, with fewer very expensive cars.")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="FUEL", order=df["FUEL"].value_counts().index)
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()
print("Observation: Petrol and diesel make up most of the listings.")

# Bivariate analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="CAR AGE", y="SELLING PRICE", alpha=0.5)
plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()
print("Observation: Older cars generally have lower selling prices.")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="KM DRIVEN", y="SELLING PRICE", alpha=0.5)
plt.xlim(0, df["KM DRIVEN"].quantile(0.99))
plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()
print("Observation: Cars with higher kilometres driven usually have lower prices, although the relationship varies.")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="TRANSMISSION", y="SELLING PRICE", showfliers=False)
plt.title("Selling Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()
print("Observation: Automatic cars have a higher typical selling price than manual cars in this dataset.")

brand_price = df.groupby("BRAND")["SELLING PRICE"].agg(["mean", "count"])
brand_price = brand_price[brand_price["count"] >= 20].sort_values("mean", ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=brand_price["mean"] / 100000, y=brand_price.index)
plt.title("Top Brands by Average Selling Price")
plt.xlabel("Average Price (Lakh INR)")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()
print("Observation: Average resale price differs clearly across brands.")

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="OWNER", y="SELLING PRICE", showfliers=False)
plt.title("Selling Price by Ownership")
plt.xlabel("Owner Type")
plt.ylabel("Selling Price")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
print("Observation: First-owner cars generally have stronger resale prices than cars with more previous owners.")

# Multivariate analysis
corr = df[["YEAR", "SELLING PRICE", "KM DRIVEN", "CAR AGE"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
print("Observation: Year and car age have a clearer relationship with selling price than kilometres driven.")

# Final findings
print("\nFinal findings")
print("1. Most used cars are concentrated in the lower and middle price range.")
print("2. Petrol and diesel are the most common fuel types.")
print("3. Older cars generally have lower selling prices.")
print("4. Higher kilometres driven is generally associated with lower resale price.")
print("5. Automatic cars have higher typical prices than manual cars in this dataset.")
print("6. Brand and ownership history also show clear differences in selling price.")
