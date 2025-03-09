import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate recent sales data similar to train.csv structure
np.random.seed(42)

# Create a date range for recent data (last 30 days)
end_date = datetime(2018, 1, 31)  # Assuming training data goes up to 2017-12-31
start_date = end_date - timedelta(days=29)
dates = pd.date_range(start=start_date, end=end_date)

# Define stores and items
stores = list(range(1, 11))  # 10 stores
items = list(range(1, 51))  # 50 items

# Generate a large dataset
data = []
for date in dates:
    for store in stores:
        for item in items:
            sales = np.random.randint(0, 300)  # Higher sales range for more realistic data
            data.append([date.strftime("%Y-%m-%d"), store, item, sales])

# Create DataFrame
recent_sales_df = pd.DataFrame(data, columns=["date", "store", "item", "sales"])

# Save to CSV
recent_sales_df.to_csv("data/recent_sales.csv", index=False)

print("✅ Large recent_sales.csv file generated!")