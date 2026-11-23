import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/bagassenop/cardekho/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# remove duplicate rows
df = df.drop_duplicates()

# make two useful columns for analysis
df["brand"] = df["name"].str.split().str[0]
df["car_age"] = df["year"].max() - df["year"]

print("\nRows after cleaning:", len(df))
print("\nNumerical summary:\n", df[["year", "selling_price", "km_driven", "car_age"]].describe())
print("\nFuel types:\n", df["fuel"].value_counts())
print("\nTransmission:\n", df["transmission"].value_counts())
print("\nOwners:\n", df["owner"].value_counts())

# price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"] / 100000, bins=30, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (Lakh INR)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# fuel type
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="fuel")
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# age and price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="car_age", y="selling_price", alpha=0.5)
plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()

# kilometres driven and price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.5)
plt.xlim(0, df["km_driven"].quantile(0.99))
plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()

# transmission and price
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="transmission", y="selling_price", showfliers=False)
plt.title("Selling Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()

# average price by brand
brand_price = df.groupby("brand")["selling_price"].agg(["mean", "count"])
brand_price = brand_price[brand_price["count"] >= 20].sort_values("mean", ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=brand_price["mean"] / 100000, y=brand_price.index)
plt.title("Top Brands by Average Selling Price")
plt.xlabel("Average Price (Lakh INR)")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()

# ownership and price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="owner", y="selling_price", showfliers=False)
plt.title("Selling Price by Ownership")
plt.xlabel("Owner Type")
plt.ylabel("Selling Price")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# correlation
corr = df[["year", "selling_price", "km_driven", "car_age"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nMain findings")
print("1. Most cars are in the lower and middle price range.")
print("2. Petrol and diesel are the most common fuel types.")
print("3. Older cars generally have lower selling prices.")
print("4. Cars driven for more kilometres usually have lower prices.")
print("5. Automatic cars are generally priced higher than manual cars in this data.")
print("6. Brand and ownership also affect the selling price.")
