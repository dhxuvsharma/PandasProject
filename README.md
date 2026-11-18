# What drives Airbnb prices in the City of Los Angeles?

BANA 212 final project, Team B4: Ashley Yang, Dhruv Sharma, Tina Tsai, Christine Lau.

**Live app:** _add your Streamlit Community Cloud link here_

We study nightly asking prices for 17,678 Airbnb listings in the City of Los Angeles (Inside Airbnb snapshot),
asking how property size, room type, amenities, host status and neighborhood relate to price. The Streamlit app
presents the analysis in four tabs:

1. **Executive Summary**: the data, why it matters, our approach and findings, in plain language.
2. **Descriptive Analytics**: the key charts from the notebook, each with a short explanation.
3. **Model Performance**: test-set comparison of seven models (our four nested OLS regressions plus tuned Ridge,
   Random Forest and Gradient Boosting), predicted-vs-actual plots and the best hyperparameters.
4. **Interactive Prediction**: pick a model, set a few listing features (the rest stay at their averages) and get a
   predicted nightly price.

The sidebar filters (room type, price, bedrooms, neighborhood, Superhost, pool) drive every descriptive chart,
including an interactive price map and a neighborhood explorer. The prediction tab adds "what if" charts that vary
one feature (bedrooms, guests, bathrooms or neighborhood) and compare all seven models.

## Repository contents

| Path | What it is |
|---|---|
| `BANA212_Airbnb_Final_Project.ipynb` | The full analysis notebook (EDA, cleaning, ANOVA, regressions) |
| `app.py` | The Streamlit app. It only loads saved files; nothing is retrained |
| `train_models.py` | Repeats the notebook's cleaning, fits and tunes the models, and saves them |
| `models/*.joblib` | The seven saved model pipelines |
| `artifacts/` | Test-set metrics, best hyperparameters, regression coefficients, test predictions |
| `data/price_data.csv.gz` | Cleaned pricing sample used for the app's charts |
| `data/listings.csv.gz` | Raw Inside Airbnb Los Angeles listings (input to `train_models.py`) |
| `requirements.txt` | Pinned package versions |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the saved models (about 3 minutes on a laptop):

```bash
python train_models.py --data data/listings.csv.gz
```

## Method notes

- Target: natural log of nightly price; predictions are converted back to dollars.
- 80/20 train/test split (`random_state=42`); hyperparameters chosen by 5-fold cross-validation on the training set.
- Best model: Gradient Boosting, test R² 0.81 on log price, median absolute error about 26% of the actual price.
- Results are associations in observational data, not causal effects.
