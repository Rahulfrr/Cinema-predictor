# Box Office & Entertainment Intelligence: Movie Hit/Flop Predictor

## Project Overview
Built an ML-powered system to predict whether a movie will achieve critical success (rating ≥ 8.0) based on production features like runtime, audience engagement, and critical scores.

## Business Problem
Entertainment studios need to understand which factors drive critical acclaim. This model identifies key predictors of movie success, enabling data-driven decisions on production planning.

## Dataset
- **Source:** IMDB Top 1000 Movies
- **Size:** 1,000 movies
- **Time Period:** 1920-2020
- **Key Features:** Release year, runtime, audience votes, critic scores, genre

## Model Performance
- **Test Accuracy:** 72%
- **Precision (Hit Detection):** 72.3%
- **Recall (Hit Detection):** 64.5%
- **F1-Score:** 0.682

### Confusion Matrix (Test Set)

Predicted Flop    Predicted Hit
Actual Flop          84                23
Actual Hit           33                60

## Key Findings
1. **Audience Engagement (No_of_Votes)** is the strongest predictor (35% importance)
   - Movies with higher voter participation tend to be rated higher
   
2. **Release Year** (23% importance) suggests that recent movies are evaluated differently
   
3. **Runtime Distribution**
   - Hit movies average **129 minutes**
   - Flop movies average **118 minutes**
   - Longer films tend to achieve higher ratings
   
4. **Genre Performance**
   - Drama dominates with 146 out of 289 movies achieving hit status (50%)
   - Action has 98 hits out of 172 (57% hit rate)
   - Comedy has lower hit rate (63 out of 155 = 41%)

## Technical Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn
- **Model:** Random Forest Classifier (100 trees, max_depth=10)
- **Data Processing:** Pandas for cleaning and feature engineering

## Project Structure

cinema_predictor/

├── imdb_top_1000.csv                    (Original dataset)

├── imdb_cleaned_hit_flop.csv            (Cleaned data with target)

├── 1_data_exploration.py                (EDA)

├── 2_data_preprocessing.py              (Data cleaning & feature engineering)

├── 3_model_training.py                  (ML model training & evaluation)

├── 4_visualizations.py                  (Dashboard creation)

├── hit_flop_model.pkl                   (Trained model)

├── movie_analysis_dashboard.png         (Analysis visualizations)

├── feature_importance.png               (Feature importance chart)

└── README.md                            (This file)



## How to Run
1. Ensure Python 3.7+ and required libraries installed
2. Place `imdb_top_1000.csv` in project directory
3. Run scripts in order: `1_data_exploration.py` → `2_data_preprocessing.py` → `3_model_training.py` → `4_visualizations.py`
4. View generated dashboards and model performance metrics

## Future Improvements
- Incorporate OTT platform data (Netflix, Amazon Prime viewership)
- Add social media sentiment analysis
- Expand to include budget and revenue predictions
- Use deep learning for NLP on movie overviews
- Real-time prediction API for upcoming movie releases

## Author
Chintha Rahul Raj
