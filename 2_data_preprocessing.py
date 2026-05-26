import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('imdb_top_1000.csv')

print(f"Original dataset: {len(df)} rows")

# Create target variable: Hit (rating >= 8.0) vs Flop (rating < 8.0)
df['Target'] = df['IMDB_Rating'].apply(lambda x: 'Hit' if x >= 8.0 else 'Flop')

print(f"\nTarget distribution:")
print(df['Target'].value_counts())
print(f"Hit percentage: {(df['Target'] == 'Hit').sum() / len(df) * 100:.1f}%")

# Create primary genre (first genre from comma-separated list)
df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip()

print(f"\nUnique Primary Genres: {df['Primary_Genre'].nunique()}")
print(df['Primary_Genre'].value_counts().head(10))

# Convert Runtime to numeric (remove 'min')
df['Runtime_mins'] = df['Runtime'].str.replace(' min', '').astype(int)

# Check data types
print(f"\nData overview:")
print(df[['Series_Title', 'Released_Year', 'Runtime_mins', 'IMDB_Rating', 'Primary_Genre', 'Target']].head(10))

# Save cleaned data
df.to_csv('imdb_cleaned_hit_flop.csv', index=False)
print("\nCleaned data saved as 'imdb_cleaned_hit_flop.csv'")