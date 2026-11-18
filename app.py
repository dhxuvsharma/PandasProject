"""Streamlit app for the BANA 212 Airbnb (City of Los Angeles) project.

Loads saved models and precomputed results; nothing is trained here.
To rebuild the saved files, run train_models.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"
AMENITIES = ["wifi", "parking", "pool_spa", "air_conditioning", "kitchen", "washer_dryer"]
LABELS = {
    "wifi": "Wi-Fi", "parking": "Parking", "pool_spa": "Pool / spa",
    "air_conditioning": "Air conditioning", "kitchen": "Kitchen", "washer_dryer": "Washer / dryer",
    "superhost": "Superhost", "availability_365": "Days available (per year)",
    "bedrooms": "Bedrooms", "bathrooms_num": "Bathrooms", "accommodates": "Guest capacity",
    "room_type": "Room type", "neighbourhood_cleansed": "Neighborhood",
}

st.set_page_config(page_title="LA Airbnb Prices", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_meta():
    with open(ROOT / "artifacts" / "metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "price_data.csv.gz")


@st.cache_data
def load_preds():
    return pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")


@st.cache_resource
def load_models(files):
    return {name: joblib.load(ROOT / "models" / fname) for name, fname in files.items()}


meta = load_meta()
df = load_data()
preds = load_preds()
models = load_models(meta["model_files"])
county = meta["county"]
metrics = meta["test_metrics"]
best_model = max(metrics, key=lambda m: metrics[m]["R2 (log price)"])
ols_full = "OLS 4: + Neighborhood"


def money(x):
    return f"${x:,.0f}"


def style(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.22, x=0, yanchor="top", title_text=""))
    return fig


def caption(text):
    st.caption(text)
    st.write("")


# ---------------------------------------------------------------- header
st.title("What drives Airbnb prices in the City of Los Angeles?")
st.markdown(
    "BANA 212 final project · Team B4: Ashley Yang, Dhruv Sharma, Tina Tsai, Christine Lau · "
    "Data: Inside Airbnb, Los Angeles listings"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Interactive Prediction"]
)

# ================================================================ TAB 1
with tab1:
    coefs = {r["term"]: r for r in meta["full_ols_coefficients"]}
    pct = lambda t: 100 * np.expm1(coefs[t]["coef"])
    bm = metrics[best_model]
    r2_in = meta["insample_ols"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings analyzed", f"{county['n_price_sample']:,}")
    c2.metric("Median nightly price", money(df["price"].median()))
    c3.metric("Best model", best_model)
    c4.metric("Typical prediction error", f"{bm['Median abs. % error']:.0f}%")

    st.subheader("The data and what we predict")
    st.write(
        f"We used a public snapshot of every Airbnb listing in Los Angeles County from Inside Airbnb: "
        f"{county['n_listings']:,} listings described by {county['n_columns']} fields such as the type of "
        f"space, its size, the amenities it advertises, information about the host, and guest ratings. "
        f"We kept only the {county['n_city']:,} listings inside the City of Los Angeles, and of those the "
        f"{county['n_price_sample']:,} that show a nightly price and have all the details we needed. "
        f"The thing we try to explain and predict is the **nightly asking price** a host advertises. "
        f"We also looked at guest star ratings as a secondary question."
    )

    st.subheader("Why it matters")
    st.write(
        "Hosts have to choose a price with little guidance: set it too high and the calendar stays empty, "
        "too low and they leave money on the table. Travelers want to know whether a listing is a fair deal, "
        "and the city has an interest in understanding what kind of short-term rental supply exists in which "
        "neighborhoods. Knowing which features actually move the price, and by how much, turns guesswork into "
        "a reasonable starting point for all three groups."
    )

    st.subheader("Our approach")
    st.write(
        "We cleaned the raw listings (for example turning text such as \"1.5 shared baths\" into a number and "
        "grouping hundreds of amenity labels into six categories), then explored how price varies with each "
        "characteristic. Because a few luxury homes cost many thousands of dollars a night, we modeled the "
        "logarithm of price, which means effects read as percentages rather than flat dollar amounts. We built "
        "the analysis up in steps: a regression on the size and type of the property, then adding amenities, "
        "then host information, then neighborhood. Finally we tested three machine-learning models (Ridge "
        "regression, a random forest and gradient boosting), tuning each one, and compared every model on "
        f"{meta['n_test']:,} listings that were held back and never used for fitting."
    )

    st.subheader("What we found")
    st.write(
        f"**Size and room type matter most.** On their own they explain about "
        f"{100 * r2_in['OLS 1: Property']['R2']:.0f}% of the differences in price. Each extra bedroom goes with "
        f"roughly {100 * np.expm1(0.1478):.0f}% higher price, and a private room costs about half what a "
        f"comparable entire home does. **Location is the next biggest factor**: adding the neighborhood lifts "
        f"the explained share to about {100 * r2_in[ols_full]['R2']:.0f}%, and the typical listing in Hollywood "
        f"Hills West asks several times what one near Exposition Park does. **A few amenities carry a premium**: "
        f"with everything else held equal, a pool or hot tub goes with about {pct('pool_spa'):.0f}% higher "
        f"prices and air conditioning about {pct('air_conditioning'):.0f}% higher, while Superhosts charge "
        f"about {pct('superhost'):.0f}% more. Guest ratings, by contrast, are almost all close to five stars "
        f"and are only weakly related to price. The best predictive model, {best_model}, explains about "
        f"{100 * bm['R2 (log price)']:.0f}% of the price differences in listings it had never seen, and its "
        f"typical miss is about {bm['Median abs. % error']:.0f}% of the actual price. These are associations "
        f"in observational data, not proof that adding a feature would cause a price change."
    )

# ================================================================ TAB 2
with tab2:
    st.header("Descriptive analytics")
    st.write(f"All charts use the {len(df):,} City of Los Angeles listings in the pricing sample.")

    # 1. price distribution
    p95 = df["price"].quantile(0.95)
    med = df["price"].median()
    fig = px.histogram(df[df["price"] <= p95], x="price", nbins=60, color_discrete_sequence=[BLUE],
                       title="Distribution of nightly prices (up to the 95th percentile)")
    fig.add_vline(med, line_dash="dash", line_color=ORANGE, annotation_text=f"Median {money(med)}")
    fig.update_layout(xaxis_title="Nightly price ($)", yaxis_title="Listings", bargap=0.05)
    st.plotly_chart(style(fig), width="stretch")
    caption(
        f"Most listings ask between {money(df['price'].quantile(.25))} and {money(df['price'].quantile(.75))} "
        f"a night, with a median of {money(med)}, but the right tail is long: the mean is "
        f"{money(df['price'].mean())} and the top price is {money(df['price'].max())}. Because of that skew we "
        f"model the logarithm of price, and the chart is cut at the 95th percentile ({money(p95)}) so the "
        f"bulk of the market is readable."
    )

    # 2. room type
    rt = df.groupby("room_type")["price"].agg(Listings="size", Median="median").sort_values("Median")
    fig = px.bar(rt, x="Median", y=rt.index, orientation="h", color_discrete_sequence=[BLUE],
                 text=rt["Median"].map(money), hover_data={"Listings": True},
                 title="Median nightly price by room type")
    fig.update_layout(xaxis_title="Median nightly price ($)", yaxis_title="")
    st.plotly_chart(style(fig, 340), width="stretch")
    caption(
        f"Entire homes and apartments (median {money(rt.loc['Entire home/apt', 'Median'])}) cost about three "
        f"times as much as private rooms ({money(rt.loc['Private room', 'Median'])}), and shared rooms are the "
        f"cheapest. Hotel rooms show the highest median, but there are only "
        f"{rt.loc['Hotel room', 'Listings']} of them, so that figure is less reliable than the others."
    )

    # 3. bedrooms & capacity
    c1, c2 = st.columns(2)
    with c1:
        b = df[df["bedrooms"].between(0, 6)].groupby("bedrooms")["price"].median()
        fig = px.line(x=b.index, y=b.values, markers=True, color_discrete_sequence=[BLUE],
                      title="Median price by number of bedrooms")
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(xaxis_title="Bedrooms", yaxis_title="Median nightly price ($)")
        st.plotly_chart(style(fig, 360), width="stretch")
        caption(
            f"The median price climbs steadily with bedrooms, from {money(b.loc[1])} for one bedroom to "
            f"{money(b.loc[3])} for three and {money(b.loc[5])} for five. Studios (0 bedrooms) price about the "
            f"same as one-bedrooms, which suggests guests pay for usable space rather than a separate room."
        )
    with c2:
        a = df[df["accommodates"].between(1, 10)].groupby("accommodates")["price"].median()
        fig = px.line(x=a.index, y=a.values, markers=True, color_discrete_sequence=[BLUE],
                      title="Median price by guest capacity")
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(xaxis_title="Maximum guests", yaxis_title="Median nightly price ($)")
        st.plotly_chart(style(fig, 360), width="stretch")
        caption(
            f"A listing for two guests has a median of {money(a.loc[2])}, while one for six reaches "
            f"{money(a.loc[6])}. The line is not perfectly smooth (odd capacities often dip), because capacity "
            f"overlaps heavily with bedrooms and room type, which is why the regression looks at them together."
        )

    # 4. neighborhoods
    nb = (df.groupby("neighbourhood_cleansed")["price"].agg(Listings="size", Median="median")
            .sort_values("Listings", ascending=False).head(10).sort_values("Median"))
    fig = px.bar(nb, x="Median", y=nb.index, orientation="h", color_discrete_sequence=[BLUE],
                 text=nb["Median"].map(money), hover_data={"Listings": True},
                 title="Median price in the 10 neighborhoods with the most listings")
    fig.add_vline(med, line_dash="dash", line_color=GRAY, annotation_text=f"City median {money(med)}", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Median nightly price ($)", yaxis_title="")
    st.plotly_chart(style(fig, 440), width="stretch")
    caption(
        f"Location produces some of the biggest gaps in the data: {nb.index[-1]} has a median of "
        f"{money(nb['Median'].iloc[-1])} while {nb.index[0]} sits at {money(nb['Median'].iloc[0])}. Part of the "
        f"gap reflects what is listed there (large hillside homes versus private rooms near USC), but adding "
        f"neighborhood to the regression still raises the explained share of price variation by about 7 to 8 "
        f"percentage points."
    )

    # 5. amenities
    rows = []
    for am in AMENITIES:
        g = df.groupby(am)["price"].median()
        rows += [{"Amenity": LABELS[am], "Status": "Without", "Median": g.get(0, np.nan)},
                 {"Amenity": LABELS[am], "Status": "With", "Median": g.get(1, np.nan)}]
    amd = pd.DataFrame(rows)
    fig = px.bar(amd, x="Amenity", y="Median", color="Status", barmode="group",
                 color_discrete_map={"Without": GRAY, "With": BLUE},
                 title="Median price with and without each amenity (unadjusted)")
    fig.update_layout(yaxis_title="Median nightly price ($)", xaxis_title="", bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(style(fig), width="stretch")
    pool = amd[amd["Amenity"] == "Pool / spa"].set_index("Status")["Median"]
    caption(
        f"Every amenity group shows a higher median when present, and pool/spa has by far the largest raw gap "
        f"({money(pool['With'])} versus {money(pool['Without'])}). These raw gaps mix the amenity with the "
        f"kind of home that tends to have it, since big houses are also the ones with pools, so the next chart "
        f"shows the gaps after adjusting for everything else."
    )

    # 6. adjusted coefficients
    cf = pd.DataFrame([r for r in meta["full_ols_coefficients"] if r["term"] in AMENITIES + ["superhost"]])
    cf["Effect"] = 100 * np.expm1(cf["coef"])
    cf["lo_pct"] = 100 * np.expm1(cf["lo"])
    cf["hi_pct"] = 100 * np.expm1(cf["hi"])
    cf["Label"] = cf["term"].map(LABELS)
    cf = cf.sort_values("Effect")
    fig = go.Figure(go.Scatter(
        x=cf["Effect"], y=cf["Label"], mode="markers", marker=dict(size=10, color=BLUE),
        error_x=dict(type="data", symmetric=False, array=cf["hi_pct"] - cf["Effect"],
                     arrayminus=cf["Effect"] - cf["lo_pct"], color=BLUE, thickness=2, width=4),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.add_vline(0, line_dash="dash", line_color=GRAY)
    fig.update_layout(title="Adjusted price difference (full regression, 95% confidence interval)",
                      xaxis_title="Price difference holding other features constant (%)", yaxis_title="")
    st.plotly_chart(style(fig, 380), width="stretch")
    caption(
        f"After controlling for size, room type, host and neighborhood, pool/spa (+{cf.set_index('term').loc['pool_spa', 'Effect']:.0f}%), "
        f"air conditioning and Superhost status keep a clear positive association with price, while Wi-Fi and "
        f"parking are indistinguishable from zero because almost every listing has them. Kitchen and washer/dryer "
        f"turn negative, most likely because they mark apartment-style and longer-stay listings rather than "
        f"because guests dislike them."
    )

    # 7. superhost & ratings
    c1, c2 = st.columns(2)
    with c1:
        sh = df.groupby("superhost")["price"].median().rename({0: "Not Superhost", 1: "Superhost"})
        fig = px.bar(x=sh.index, y=sh.values, color_discrete_sequence=[BLUE], text=[money(v) for v in sh.values],
                     title="Median price by Superhost status")
        fig.update_layout(xaxis_title="", yaxis_title="Median nightly price ($)")
        st.plotly_chart(style(fig, 360), width="stretch")
        caption(
            f"Superhost listings have a median of {money(sh['Superhost'])} against {money(sh['Not Superhost'])} "
            f"for other hosts. Some of this gap comes from Superhosts running larger homes, but the adjusted "
            f"premium is still about {pct('superhost'):.0f}%."
        )
    with c2:
        rd = df.dropna(subset=["review_scores_rating"]).copy()
        rd["Price decile"] = pd.qcut(rd["price"], 10, labels=False) + 1
        rr = rd.groupby("Price decile")["review_scores_rating"].mean()
        fig = px.line(x=rr.index, y=rr.values, markers=True, color_discrete_sequence=[BLUE],
                      title="Average guest rating by price decile")
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(xaxis_title="Price decile (1 = cheapest 10%)", yaxis_title="Average rating (out of 5)",
                          xaxis=dict(dtick=1))
        st.plotly_chart(style(fig, 360), width="stretch")
        caption(
            f"Ratings drift up slightly with price, from {rr.iloc[0]:.2f} in the cheapest tenth of listings to "
            f"{rr.iloc[-1]:.2f} in the most expensive, but the rank correlation is only "
            f"{meta['rating_model']['spearman']:.2f}. Nearly all ratings sit between 4.7 and 5.0, so our ratings "
            f"regression explains only about {100 * meta['rating_model']['r2']:.0f}% of their variation."
        )

# ================================================================ TAB 3
with tab3:
    st.header("Model performance")
    st.write(
        f"All models predict the log of nightly price from the same pool of listing features. They were fit on "
        f"{meta['n_train']:,} listings (80%) and scored on the {meta['n_test']:,} held-out listings (20%). The "
        f"four OLS models are the nested regressions from our notebook; Ridge, Random Forest and Gradient "
        f"Boosting were tuned with cross-validation on the training set only."
    )

    tbl = pd.DataFrame(metrics).T.astype(float)
    tbl.insert(0, "In-sample R² (notebook)",
               [float(meta["insample_ols"][m]["R2"]) if m in meta["insample_ols"] else np.nan for m in tbl.index])
    tbl = tbl[["Features", "In-sample R² (notebook)", "R2 (log price)", "RMSE (log price)", "MAE ($)",
               "Median abs. error ($)", "Median abs. % error"]]
    tbl = tbl.rename(columns={"R2 (log price)": "Test R²", "RMSE (log price)": "Test RMSE (log)",
                              "MAE ($)": "Test MAE ($)", "Median abs. error ($)": "Median abs. error ($)",
                              "Median abs. % error": "Median abs. error (%)"})
    st.subheader("Comparison table (held-out test set)")
    st.dataframe(
        tbl.style.format({"Features": "{:.0f}", "In-sample R² (notebook)": "{:.3f}", "Test R²": "{:.3f}",
                          "Test RMSE (log)": "{:.3f}", "Test MAE ($)": "${:,.0f}",
                          "Median abs. error ($)": "${:,.0f}", "Median abs. error (%)": "{:.1f}%"},
                         na_rep="–")
        .highlight_max(subset=["Test R²"], color="#2a78d633")
        .highlight_min(subset=["Test RMSE (log)", "Test MAE ($)", "Median abs. error (%)"], color="#2a78d633"),
        width="stretch")
    st.caption(
        "Features counts input variables (neighborhood and room type each count as one). In-sample R² is from "
        "the notebook's full-sample regressions and applies only to the OLS models."
    )

    c1, c2 = st.columns(2)
    order = list(tbl.index)
    colors = [ORANGE if m == best_model else BLUE for m in order]
    with c1:
        fig = go.Figure(go.Bar(x=tbl["Test R²"], y=order, orientation="h", marker_color=colors,
                               text=tbl["Test R²"].map("{:.3f}".format), textposition="outside"))
        fig.update_layout(title="Test R² (higher is better)", xaxis=dict(range=[0, 1]),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 380), width="stretch")
    with c2:
        fig = go.Figure(go.Bar(x=tbl["Median abs. error (%)"], y=order, orientation="h", marker_color=colors,
                               text=tbl["Median abs. error (%)"].map("{:.1f}%".format), textposition="outside"))
        fig.update_layout(title="Median absolute % error (lower is better)",
                          xaxis=dict(range=[0, tbl["Median abs. error (%)"].max() * 1.2]),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 380), width="stretch")
    st.caption(
        f"Each step of the nested OLS models helps, with neighborhood giving the largest single jump. Ridge "
        f"matches the full OLS model almost exactly, which shows the regression was not overfitting. The tree "
        f"models do best because they can capture interactions (for example, a pool is worth more in the hills "
        f"than in a studio), and {best_model} (orange) is the overall winner."
    )

    st.subheader("Predicted vs. actual price")
    pick = st.selectbox("Model", order, index=order.index(best_model), key="pva")
    pv = pd.DataFrame({"Actual": np.exp(preds["actual_log"]), "Predicted": np.exp(preds[pick]),
                       "Room type": preds["room_type"]})
    lo, hi = 10, max(pv["Actual"].max(), pv["Predicted"].max())
    fig = px.scatter(pv, x="Actual", y="Predicted", log_x=True, log_y=True, opacity=0.35,
                     color_discrete_sequence=[BLUE], hover_data=["Room type"],
                     title=f"{pick}: predicted vs. actual nightly price (test set, log scale)")
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=ORANGE, dash="dash", width=2),
                             name="Perfect prediction"))
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
    ticks = [10, 30, 100, 300, 1000, 3000, 10000]
    fig.update_layout(xaxis_title="Actual price ($)", yaxis_title="Predicted price ($)",
                      xaxis=dict(tickvals=ticks), yaxis=dict(tickvals=ticks))
    st.plotly_chart(style(fig, 520), width="stretch")
    st.caption(
        "Points on the dashed line are perfect predictions. The cloud follows the line across the whole range "
        "from $20 rooms to $5,000+ homes, but it fans out at the top, where a handful of luxury listings are "
        "hard for any model to price."
    )

    st.subheader("Best hyperparameters")
    for m in order:
        params = {("CV RMSE, log price (score, not a parameter)" if k == "CV RMSE (log)" else k): v
                  for k, v in meta["best_params"][m].items()}
        if "(none)" in params:
            continue
        st.markdown(f"**{m}**")
        st.dataframe(pd.DataFrame({"Hyperparameter": list(params), "Best value": [str(v) for v in params.values()]}),
                     hide_index=True, width="stretch")
    st.caption(
        "Ridge was tuned with a 5-fold grid search over the penalty alpha; Random Forest (12 candidates) and "
        "Gradient Boosting (20 candidates) with 5-fold randomized search, all scored by RMSE on log price. "
        "The four OLS models have no hyperparameters."
    )

# ================================================================ TAB 4
with tab4:
    st.header("Interactive prediction")
    st.write(
        "Pick a model and describe a listing. Any feature you leave on **Average** is held at the average of "
        "the City of Los Angeles listings, so you can see the effect of the features you change."
    )
    d = meta["defaults"]
    left, right = st.columns([1, 1.3])
    with left:
        model_name = st.selectbox("Model", order, index=order.index(best_model))
        room = st.selectbox("Room type", meta["room_types"], index=meta["room_types"].index(d["room_type"]))
        hood = st.selectbox("Neighborhood", meta["neighbourhoods"],
                            index=meta["neighbourhoods"].index(d["neighbourhood_cleansed"]))
        c1, c2, c3 = st.columns(3)
        beds = c1.number_input("Bedrooms", 0, 10, int(round(d["bedrooms"])))
        baths = c2.number_input("Bathrooms", 0.0, 10.0, float(round(d["bathrooms_num"] * 2) / 2), step=0.5)
        guests = c3.number_input("Guests", 1, 16, int(round(d["accommodates"])))

        def tri(label, key):
            opt = st.radio(label, [f"Average ({100 * d[key]:.0f}% have it)", "Yes", "No"], horizontal=True,
                           key=f"r_{key}")
            return d[key] if opt.startswith("Average") else (1.0 if opt == "Yes" else 0.0)

        sh = tri("Superhost", "superhost")
        pool = tri("Pool / spa", "pool_spa")
        ac = tri("Air conditioning", "air_conditioning")

    row = {k: d[k] for k in ["availability_365"] + AMENITIES}
    row.update({"room_type": room, "neighbourhood_cleansed": hood, "bedrooms": beds, "bathrooms_num": baths,
                "accommodates": guests, "superhost": sh, "pool_spa": pool, "air_conditioning": ac})
    X = pd.DataFrame([row])

    def predict(name):
        m = models[name]
        return float(np.exp(m["pipeline"].predict(X[m["features"]])[0]))

    with right:
        price = predict(model_name)
        st.metric(f"Predicted nightly price ({model_name})", money(price))
        err = metrics[model_name]["Median abs. % error"]
        st.caption(
            f"On held-out listings this model's typical miss was about {err:.0f}%, so read this as roughly "
            f"{money(price * (1 - err / 100))} to {money(price * (1 + err / 100))}. Prices come back from the log "
            f"scale, so they sit closer to a typical (median) price than an average one."
        )
        if any(f not in models[model_name]["features"] for f in ["neighbourhood_cleansed", "superhost", "pool_spa"]):
            unused = [LABELS[f] for f in ["neighbourhood_cleansed", "superhost", "pool_spa", "air_conditioning"]
                      if f not in models[model_name]["features"]]
            st.info(f"This model does not use: {', '.join(unused)}. Changing those inputs has no effect here.")

        allp = pd.DataFrame({"Model": order, "Prediction": [predict(m) for m in order]})
        fig = go.Figure(go.Bar(x=allp["Prediction"], y=allp["Model"], orientation="h",
                               marker_color=[ORANGE if m == model_name else BLUE for m in order],
                               text=allp["Prediction"].map(money), textposition="outside"))
        fig.update_layout(title="Same listing, every model", xaxis_title="Predicted nightly price ($)",
                          yaxis=dict(autorange="reversed"),
                          xaxis=dict(range=[0, allp["Prediction"].max() * 1.25]))
        st.plotly_chart(style(fig, 360), width="stretch")

        with st.expander("Features held at their averages"):
            held = {LABELS[k]: (f"{v:.0f} days" if k == "availability_365" else f"{100 * v:.0f}% of listings")
                    for k, v in d.items() if k in ["availability_365", "wifi", "parking", "kitchen", "washer_dryer"]}
            st.table(pd.Series(held, name="Value used"))

st.divider()
st.caption("Associations from observational listing data, not causal effects. Models are loaded from saved files; "
           "nothing is retrained in this app.")
