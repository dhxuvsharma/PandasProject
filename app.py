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
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).parent

# ------------------------------------------------------------------ palette
INK, MUTED, GRID = "#1F2333", "#6B7085", "#E6E8EF"
CORAL, NAVY, TEAL, GOLD, GRAY = "#E8505B", "#2A3F8F", "#14A38B", "#F2A541", "#B9BCC8"
SEQ = ["#FDE6E7", "#F7A8AD", "#E8505B", "#B8323C", "#7A1C24"]
ROOM_COLORS = {"Entire home/apt": NAVY, "Private room": CORAL, "Hotel room": GOLD, "Shared room": TEAL}

AMENITIES = ["wifi", "parking", "pool_spa", "air_conditioning", "kitchen", "washer_dryer"]
LABELS = {
    "wifi": "Wi-Fi", "parking": "Parking", "pool_spa": "Pool / spa",
    "air_conditioning": "Air conditioning", "kitchen": "Kitchen", "washer_dryer": "Washer / dryer",
    "superhost": "Superhost", "availability_365": "Days available (per year)",
    "bedrooms": "Bedrooms", "bathrooms_num": "Bathrooms", "accommodates": "Guest capacity",
    "room_type": "Room type", "neighbourhood_cleansed": "Neighborhood",
}

pio.templates["la"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[NAVY, CORAL, TEAL, GOLD, "#7B61C9", "#3A8DDE"],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                   title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                   title_font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, family="Inter, sans-serif")),
        legend=dict(orientation="h", y=-0.2, x=0, yanchor="top", title_text=""),
        margin=dict(l=10, r=20, t=50, b=10),
        bargap=0.25,
    )
)
pio.templates.default = "plotly_white+la"

st.set_page_config(page_title="LA Airbnb Prices", page_icon="🏡", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }

.hero {
  background: linear-gradient(120deg, #1F2A5C 0%, #2A3F8F 55%, #E8505B 140%);
  border-radius: 22px; padding: 34px 38px 30px; color: white; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(31,42,92,.18);
}
.hero .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 12px; opacity: .8; font-weight: 600; }
.hero h1 { color: white; font-size: 40px; font-weight: 800; margin: 6px 0 8px; line-height: 1.1; padding: 0; }
.hero p { color: rgba(255,255,255,.85); font-size: 15px; margin: 0; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin-top: 24px; }
.kpi { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); border-radius: 14px; padding: 14px 16px; }
.kpi .v { font-size: 26px; font-weight: 800; color: white; line-height: 1.1; }
.kpi .l { font-size: 12px; color: rgba(255,255,255,.78); margin-top: 4px; }
@media (max-width: 800px) { .kpis { grid-template-columns: repeat(2, minmax(0,1fr)); } .hero h1 { font-size: 28px; } }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: white; padding: 6px; border-radius: 14px;
  border: 1px solid #E6E8EF; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 18px; border-radius: 10px; font-weight: 600; color: #6B7085; }
.stTabs [aria-selected="true"] { background: #1F2A5C !important; color: white !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

div[data-testid="stVerticalBlockBorderWrapper"] { background: white; border-radius: 16px !important;
  border: 1px solid #E6E8EF !important; box-shadow: 0 2px 10px rgba(31,35,51,.04); }

.card-h { display:flex; align-items:center; gap:10px; margin-bottom: 4px; }
.card-h .ico { width: 36px; height: 36px; border-radius: 10px; display:flex; align-items:center; justify-content:center;
  font-size: 18px; background: #FDE6E7; }
.card-h h3 { margin: 0; font-size: 19px; font-weight: 700; padding: 0; }
.note { color: #6B7085; font-size: 14px; line-height: 1.55; margin-top: -4px; }
.big-price { font-size: 56px; font-weight: 800; color: #1F2A5C; letter-spacing: -0.03em; line-height: 1; }
.range { color: #6B7085; font-size: 14px; margin-top: 6px; }
.chip { display:inline-block; background:#F1F2F7; color:#1F2333; border-radius:999px; padding:4px 10px;
  font-size:12px; font-weight:600; margin:2px 4px 2px 0; }
.chip.win { background:#FDE6E7; color:#B8323C; }
.finding { border-left: 4px solid #E8505B; padding: 6px 0 6px 14px; margin: 8px 0; }
.finding b { color: #1F2A5C; }
section[data-testid="stSidebar"] { background: white; border-right: 1px solid #E6E8EF; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
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
df_all = load_data()
preds = load_preds()
models = load_models(meta["model_files"])
county = meta["county"]
metrics = meta["test_metrics"]
order = list(metrics)
best_model = max(metrics, key=lambda m: metrics[m]["R2 (log price)"])
ols_full = "OLS 4: + Neighborhood"
coefs = {r["term"]: r for r in meta["full_ols_coefficients"]}
pct = lambda t: 100 * np.expm1(coefs[t]["coef"])
d = meta["defaults"]


def money(x):
    return f"${x:,.0f}"


def chart(fig, height=400, key=None):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def card_header(icon, title, sub=None):
    st.markdown(f'<div class="card-h"><div class="ico">{icon}</div><h3>{title}</h3></div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="note">{sub}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.markdown("### 🔎 Filter the listings")
    st.caption("Filters update every chart in **Descriptive Analytics**.")
    rooms_sel = st.multiselect("Room type", meta["room_types"], default=meta["room_types"])
    pmax = int(np.ceil(df_all["price"].quantile(0.99) / 50) * 50)
    price_rng = st.slider("Nightly price ($)", 0, pmax, (0, pmax), step=25,
                          help="Upper end includes everything above it, so luxury listings stay in.")
    bed_rng = st.slider("Bedrooms", 0, 8, (0, 8), help="8 includes 8 or more.")
    top_hoods = df_all["neighbourhood_cleansed"].value_counts().index.tolist()
    hoods_sel = st.multiselect("Neighborhoods", top_hoods, placeholder="All neighborhoods")
    sh_only = st.toggle("Superhosts only")
    pool_only = st.toggle("Pool / spa only")

df = df_all[df_all["room_type"].isin(rooms_sel)]
hi_price = np.inf if price_rng[1] >= pmax else price_rng[1]
df = df[(df["price"] >= price_rng[0]) & (df["price"] <= hi_price)]
hi_bed = np.inf if bed_rng[1] >= 8 else bed_rng[1]
df = df[(df["bedrooms"] >= bed_rng[0]) & (df["bedrooms"] <= hi_bed)]
if hoods_sel:
    df = df[df["neighbourhood_cleansed"].isin(hoods_sel)]
if sh_only:
    df = df[df["superhost"] == 1]
if pool_only:
    df = df[df["pool_spa"] == 1]
with st.sidebar:
    st.metric("Listings matching", f"{len(df):,}", f"{len(df) - len(df_all):,} vs all" if len(df) < len(df_all) else None)

# ------------------------------------------------------------------ hero
bm = metrics[best_model]
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">BANA 212 · Team B4 · Inside Airbnb data</div>
  <h1>What drives Airbnb prices<br>in the City of Los Angeles?</h1>
  <p>Ashley Yang · Dhruv Sharma · Tina Tsai · Christine Lau</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{county['n_price_sample']:,}</div><div class="l">listings analyzed</div></div>
    <div class="kpi"><div class="v">{money(df_all['price'].median())}</div><div class="l">median nightly price</div></div>
    <div class="kpi"><div class="v">{100 * bm['R2 (log price)']:.0f}%</div><div class="l">of price variation explained ({best_model})</div></div>
    <div class="kpi"><div class="v">±{bm['Median abs. % error']:.0f}%</div><div class="l">typical prediction error</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋  Executive Summary", "📊  Descriptive Analytics", "🏆  Model Performance", "🔮  Interactive Prediction"]
)

# ================================================================ TAB 1
with tab1:
    r2_in = meta["insample_ols"]
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            card_header("🗂️", "The data and what we predict")
            st.write(
                f"We used a public snapshot of every Airbnb listing in Los Angeles County from Inside Airbnb: "
                f"{county['n_listings']:,} listings described by {county['n_columns']} fields such as the type of "
                f"space, its size, the amenities it advertises, information about the host, and guest ratings. "
                f"We kept only the {county['n_city']:,} listings inside the City of Los Angeles, and of those the "
                f"{county['n_price_sample']:,} that show a nightly price and have all the details we needed. "
                f"The thing we try to explain and predict is the **nightly asking price** a host advertises; "
                f"guest star ratings are a secondary question."
            )
        with st.container(border=True):
            card_header("💡", "Why it matters")
            st.write(
                "Hosts have to choose a price with little guidance: set it too high and the calendar stays empty, "
                "too low and they leave money on the table. Travelers want to know whether a listing is a fair "
                "deal, and the city has an interest in understanding what kind of short-term rental supply exists "
                "in which neighborhoods. Knowing which features actually move the price, and by how much, turns "
                "guesswork into a reasonable starting point for all three groups."
            )
        with st.container(border=True):
            card_header("🧭", "Our approach")
            st.write(
                "We cleaned the raw listings (for example turning text such as \"1.5 shared baths\" into a number "
                "and grouping hundreds of amenity labels into six categories), then explored how price varies with "
                "each characteristic. Because a few luxury homes cost thousands of dollars a night, we modeled the "
                "logarithm of price, so effects read as percentages rather than flat dollar amounts. We built the "
                "analysis up in steps: size and type of the property, then amenities, then host information, then "
                "neighborhood. Finally we tuned three machine-learning models (Ridge regression, a random forest and "
                f"gradient boosting) and compared all seven models on {meta['n_test']:,} listings held back from "
                "fitting."
            )
        with st.container(border=True):
            card_header("🎯", "What we found")
            st.write(
                f"Size and room type matter most: on their own they explain about "
                f"{100 * r2_in['OLS 1: Property']['R2']:.0f}% of the differences in price. Location is next, lifting "
                f"that share to about {100 * r2_in[ols_full]['R2']:.0f}%. With everything else held equal, a pool or "
                f"hot tub goes with about {pct('pool_spa'):.0f}% higher prices, air conditioning about "
                f"{pct('air_conditioning'):.0f}% higher, and Superhosts charge about {pct('superhost'):.0f}% more. "
                f"Guest ratings are almost all near five stars and only weakly tied to price. Our best model, "
                f"{best_model}, explains about {100 * bm['R2 (log price)']:.0f}% of price differences on listings it "
                f"never saw, typically missing by about {bm['Median abs. % error']:.0f}%. These are associations in "
                f"observational data, not proof that adding a feature would cause a price change."
            )
    with right:
        with st.container(border=True):
            card_header("⚡", "Findings at a glance")
            st.markdown(f"""
<div class="finding"><b>+{100 * np.expm1(0.1478):.0f}% per bedroom</b><br><span class="note">holding guests, bathrooms and room type fixed</span></div>
<div class="finding"><b>Private room ≈ half price</b><br><span class="note">about {abs(100 * np.expm1(-0.67)):.0f}% less than a comparable entire home</span></div>
<div class="finding"><b>Location adds {100 * (r2_in[ols_full]['R2'] - r2_in['OLS 3: + Host']['R2']):.0f} pts of R²</b><br><span class="note">Hollywood Hills West vs Exposition Park: ~20× gap in medians</span></div>
<div class="finding"><b>Pool / spa +{pct('pool_spa'):.0f}%</b><br><span class="note">the biggest amenity premium after adjustment</span></div>
<div class="finding"><b>Superhost +{pct('superhost'):.0f}%</b><br><span class="note">and +{meta['rating_model']['superhost_coef']:.2f} stars in ratings</span></div>
""", unsafe_allow_html=True)
        with st.container(border=True):
            card_header("📈", "Adjusted effect on price")
            eff = pd.DataFrame([{"Feature": LABELS[t], "Effect": pct(t)}
                                for t in ["pool_spa", "air_conditioning", "superhost", "parking", "wifi",
                                          "washer_dryer", "kitchen"]]).sort_values("Effect")
            fig = go.Figure(go.Bar(x=eff["Effect"], y=eff["Feature"], orientation="h",
                                   marker_color=[CORAL if v > 0 else GRAY for v in eff["Effect"]],
                                   text=[f"{v:+.0f}%" for v in eff["Effect"]], textposition="outside",
                                   hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
            fig.update_traces(cliponaxis=False)
            fig.update_layout(xaxis_title="% change in price, all else equal", margin=dict(t=10, l=10, r=40),
                              xaxis=dict(range=[-38, 40]))
            chart(fig, 330)

# ================================================================ TAB 2
with tab2:
    if len(df) < 30:
        st.warning("Fewer than 30 listings match the sidebar filters, so the charts below show all listings.")
        df = df_all
    filt_note = "" if len(df) == len(df_all) else f" (filtered to {len(df):,} listings)"
    view = st.segmented_control(
        "Section", ["💲 Price & size", "📍 Location", "🛁 Amenities & hosts", "⭐ Ratings"],
        default="💲 Price & size", label_visibility="collapsed")
    view = view or "💲 Price & size"
    med = df["price"].median()

    if view.endswith("Price & size"):
        with st.container(border=True):
            card_header("💲", f"How nightly prices are distributed{filt_note}")
            c1, c2 = st.columns([3, 1])
            log_x = c2.toggle("Log scale", value=False, help="Spread out the cheap end and squeeze the luxury tail")
            color_rt = c2.toggle("Split by room type", value=True)
            p95 = df["price"].quantile(0.95)
            plot_df = df if log_x else df[df["price"] <= p95]
            fig = px.histogram(plot_df, x="price", nbins=70, log_x=log_x,
                               color="room_type" if color_rt else None, color_discrete_map=ROOM_COLORS,
                               color_discrete_sequence=[NAVY], barmode="stack",
                               labels={"price": "Nightly price ($)", "room_type": "Room type"})
            fig.add_vline(med, line_dash="dash", line_color=INK, annotation_text=f"Median {money(med)}",
                          annotation_font_color=INK)
            fig.update_layout(yaxis_title="Listings", bargap=0.04)
            with c1:
                chart(fig, 380)
            note(
                f"Most listings ask between {money(df['price'].quantile(.25))} and {money(df['price'].quantile(.75))} "
                f"a night, with a median of {money(med)}, but the right tail is long: the mean is "
                f"{money(df['price'].mean())} and the top price is {money(df['price'].max())}. That skew is why we "
                f"model the logarithm of price"
                + ("." if log_x else f", and why this view stops at the 95th percentile ({money(p95)}); switch on "
                   f"the log scale to see every listing.")
            )

        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            card_header("🛏️", "Median price by room type")
            rt = df.groupby("room_type")["price"].agg(Listings="size", Median="median").sort_values("Median")
            fig = go.Figure(go.Bar(x=rt["Median"], y=rt.index, orientation="h",
                                   marker_color=[ROOM_COLORS.get(r, NAVY) for r in rt.index],
                                   text=rt["Median"].map(money), textposition="outside",
                                   customdata=rt["Listings"],
                                   hovertemplate="%{y}<br>Median %{x:$,.0f}<br>%{customdata:,} listings<extra></extra>"))
            fig.update_layout(xaxis_title="Median nightly price ($)", xaxis=dict(range=[0, rt["Median"].max() * 1.25]),
                              margin=dict(t=10))
            chart(fig, 300)
            top, bot = rt.index[-1], rt.index[0]
            note(
                f"{top} listings have the highest median ({money(rt['Median'].iloc[-1])}, {rt['Listings'].iloc[-1]:,} "
                f"listings) and {bot} the lowest ({money(rt['Median'].iloc[0])}). Entire homes cost roughly three "
                f"times what private rooms do across the full sample, and hotel and shared rooms are rare, so their "
                f"medians move a lot with small changes in the data."
            )
        with c2, st.container(border=True):
            card_header("📐", "Price grows with size")
            dim = st.segmented_control("Size measure", ["Bedrooms", "Guests", "Bathrooms"], default="Bedrooms",
                                       key="dim", label_visibility="collapsed") or "Bedrooms"
            col, lim = {"Bedrooms": ("bedrooms", 6), "Guests": ("accommodates", 12),
                        "Bathrooms": ("bathrooms_num", 5)}[dim]
            g = df[df[col] <= lim].groupby(col)["price"].agg(Median="median", Q1=lambda s: s.quantile(.25),
                                                             Q3=lambda s: s.quantile(.75), n="size")
            g = g[g["n"] >= 10]
            fig = go.Figure([
                go.Scatter(x=list(g.index) + list(g.index[::-1]), y=list(g["Q3"]) + list(g["Q1"][::-1]),
                           fill="toself", fillcolor="rgba(42,63,143,.12)", line=dict(width=0), mode="lines",
                           hoverinfo="skip", name="Middle 50%"),
                go.Scatter(x=g.index, y=g["Median"], mode="lines+markers", line=dict(color=NAVY, width=3),
                           marker=dict(size=9, color=NAVY, line=dict(color="white", width=2)), name="Median",
                           customdata=g["n"],
                           hovertemplate=f"{dim}: %{{x}}<br>Median %{{y:$,.0f}}<br>%{{customdata:,}} listings<extra></extra>"),
            ])
            fig.update_layout(xaxis_title=dim, yaxis_title="Nightly price ($)", margin=dict(t=10))
            chart(fig, 300)
            note(
                f"The median price rises from {money(g['Median'].iloc[0])} at the smallest {dim.lower()} level shown "
                f"to {money(g['Median'].iloc[-1])} at {g.index[-1]:g}. The shaded band is the middle half of "
                f"listings, and it widens as homes grow, which means size alone pins down price less well for "
                f"bigger properties."
            )

    elif view.endswith("Location"):
        with st.container(border=True):
            card_header("🗺️", f"Where the listings are and what they cost{filt_note}")
            c1, c2 = st.columns([3, 1])
            with c2:
                cap = st.select_slider("Color scale cap", [300, 500, 750, 1000, 2000], value=750,
                                       format_func=lambda v: f"${v:,}",
                                       help="Prices above the cap get the darkest color")
                sample = st.toggle("Sample 5,000 points (faster)", value=len(df) > 6000)
            mdf = df.dropna(subset=["latitude", "longitude"])
            if sample and len(mdf) > 5000:
                mdf = mdf.sample(5000, random_state=1)
            fig = px.scatter_map(
                mdf, lat="latitude", lon="longitude", color=mdf["price"].clip(upper=cap),
                color_continuous_scale=SEQ, range_color=[0, cap], zoom=9.2,
                center={"lat": 34.05, "lon": -118.38}, map_style="carto-positron", opacity=0.75,
                hover_name="name", custom_data=["price", "room_type", "neighbourhood_cleansed", "bedrooms"])
            fig.update_traces(marker=dict(size=6),
                              hovertemplate="<b>%{hovertext}</b><br>%{customdata[2]}<br>%{customdata[1]}, "
                                            "%{customdata[3]} bd<br>%{customdata[0]:$,.0f} / night<extra></extra>")
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar=dict(title="$ / night", thickness=12, tickprefix="$"))
            with c1:
                chart(fig, 520)
            with c2:
                note("Zoom and pan the map, and hover any dot to see the listing. Use the sidebar to narrow "
                     "to a room type or a few neighborhoods.")
            note(
                "The darkest, most expensive listings cluster in the hills and along the coast, in Hollywood Hills, "
                "Bel Air, Malibu-adjacent areas and Venice. The cheapest cluster near downtown, around USC and in the "
                "San Fernando Valley, which is why neighborhood adds so much to the price models."
            )

        c1, c2 = st.columns([1.2, 1], gap="medium")
        with c1, st.container(border=True):
            card_header("🏘️", "Neighborhood league table")
            n_show = st.slider("Neighborhoods to show (by number of listings)", 5, 25, 12)
            nb = (df.groupby("neighbourhood_cleansed")["price"].agg(Listings="size", Median="median")
                  .sort_values("Listings", ascending=False).head(n_show).sort_values("Median"))
            fig = go.Figure(go.Bar(x=nb["Median"], y=nb.index, orientation="h",
                                   marker=dict(color=nb["Median"], colorscale=SEQ[1:]),
                                   text=nb["Median"].map(money), textposition="outside",
                                   customdata=nb["Listings"],
                                   hovertemplate="%{y}<br>Median %{x:$,.0f}<br>%{customdata:,} listings<extra></extra>"))
            fig.add_vline(med, line_dash="dot", line_color=MUTED, annotation_text=f"median {money(med)}",
                          annotation_position="bottom right", annotation_font_color=MUTED)
            fig.update_layout(xaxis_title="Median nightly price ($)", margin=dict(t=10),
                              xaxis=dict(range=[0, nb["Median"].max() * 1.2]))
            chart(fig, 120 + 28 * len(nb))
            note(
                f"Among the {len(nb)} neighborhoods with the most listings, {nb.index[-1]} has the highest median "
                f"({money(nb['Median'].iloc[-1])}) and {nb.index[0]} the lowest ({money(nb['Median'].iloc[0])}). Part of "
                f"the gap is what gets listed there, but neighborhood still adds about 7 percentage points of "
                f"explained variation after controlling for size, amenities and host."
            )
        with c2, st.container(border=True):
            card_header("🔍", "Neighborhood explorer")
            hood_counts = df["neighbourhood_cleansed"].value_counts()
            hood = st.selectbox("Pick a neighborhood", hood_counts.index.tolist(),
                                format_func=lambda h: f"{h} ({hood_counts[h]:,})")
            h = df[df["neighbourhood_cleansed"] == hood]
            m1, m2 = st.columns(2)
            m1.metric("Median price", money(h["price"].median()),
                      f"{100 * (h['price'].median() / med - 1):+.0f}% vs filtered median")
            m2.metric("Listings", f"{len(h):,}")
            m3, m4 = st.columns(2)
            m3.metric("Entire homes", f"{100 * (h['room_type'] == 'Entire home/apt').mean():.0f}%")
            m4.metric("Avg rating", f"{h['review_scores_rating'].mean():.2f} ★" if h["review_scores_rating"].notna().any() else "–")
            comp = pd.DataFrame({
                "Share": [h[a].mean() * 100 for a in AMENITIES] + [df[a].mean() * 100 for a in AMENITIES],
                "Amenity": [LABELS[a] for a in AMENITIES] * 2,
                "Group": [hood] * 6 + ["All filtered"] * 6})
            fig = px.bar(comp, x="Share", y="Amenity", color="Group", barmode="group", orientation="h",
                         color_discrete_sequence=[CORAL, GRAY], labels={"Share": "% of listings"})
            fig.update_layout(margin=dict(t=10), legend=dict(y=-0.25), yaxis_title="")
            chart(fig, 300)
            note(f"How {hood} compares with the rest of the filtered listings. Amenity bars show the share of "
                 f"listings that advertise each amenity group.")

    elif view.endswith("Amenities & hosts"):
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            card_header("🛁", "Raw price gap by amenity")
            rows = []
            for am in AMENITIES:
                gm = df.groupby(am)["price"].median()
                rows.append({"Amenity": LABELS[am], "Without": gm.get(0, np.nan), "With": gm.get(1, np.nan)})
            amd = pd.DataFrame(rows).dropna()
            amd["Gap"] = amd["With"] - amd["Without"]
            amd = amd.sort_values("Gap")
            fig = go.Figure()
            for _, r in amd.iterrows():
                fig.add_shape(type="line", x0=r["Without"], x1=r["With"], y0=r["Amenity"], y1=r["Amenity"],
                              line=dict(color=GRID, width=6))
            fig.add_trace(go.Scatter(x=amd["Without"], y=amd["Amenity"], mode="markers", name="Without",
                                     marker=dict(size=14, color=GRAY, line=dict(color="white", width=2)),
                                     hovertemplate="Without: %{x:$,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=amd["With"], y=amd["Amenity"], mode="markers", name="With",
                                     marker=dict(size=14, color=CORAL, line=dict(color="white", width=2)),
                                     hovertemplate="With: %{x:$,.0f}<extra></extra>"))
            fig.update_layout(xaxis_title="Median nightly price ($)", margin=dict(t=10))
            chart(fig, 340)
            big = amd.iloc[-1]
            note(
                f"Listings that advertise each amenity have a higher median price, and {big['Amenity'].lower()} shows "
                f"the widest raw gap ({money(big['With'])} with versus {money(big['Without'])} without). These raw gaps "
                f"mix the amenity with the kind of home that tends to have it, which the adjusted chart on the right "
                f"separates out."
            )
        with c2, st.container(border=True):
            card_header("⚖️", "Adjusted effect (full regression)")
            cf = pd.DataFrame([r for r in meta["full_ols_coefficients"] if r["term"] in AMENITIES + ["superhost"]])
            cf["Effect"] = 100 * np.expm1(cf["coef"])
            cf["lo_pct"], cf["hi_pct"] = 100 * np.expm1(cf["lo"]), 100 * np.expm1(cf["hi"])
            cf["Label"] = cf["term"].map(LABELS)
            cf["sig"] = cf["p"] < 0.05
            cf = cf.sort_values("Effect")
            fig = go.Figure(go.Scatter(
                x=cf["Effect"], y=cf["Label"], mode="markers",
                marker=dict(size=13, color=[CORAL if s else GRAY for s in cf["sig"]], line=dict(color="white", width=2)),
                error_x=dict(type="data", symmetric=False, array=cf["hi_pct"] - cf["Effect"],
                             arrayminus=cf["Effect"] - cf["lo_pct"], color=MUTED, thickness=1.5, width=5),
                customdata=cf["p"], hovertemplate="%{y}: %{x:+.1f}%<br>p = %{customdata:.3g}<extra></extra>"))
            fig.add_vline(0, line_dash="dash", line_color=MUTED)
            fig.update_layout(xaxis_title="% price difference, all else equal (95% CI)", margin=dict(t=10))
            chart(fig, 340)
            note(
                f"Holding size, room type, host and neighborhood fixed, pool/spa (+{pct('pool_spa'):.0f}%), air "
                f"conditioning and Superhost status keep clear positive effects (coral = significant at 5%). Wi-Fi and "
                f"parking are indistinguishable from zero because nearly every listing has them, and kitchen and "
                f"washer/dryer turn negative, likely because they mark apartment-style, longer-stay listings."
            )

        with st.container(border=True):
            card_header("🏅", "Superhosts and availability")
            c1, c2 = st.columns(2)
            with c1:
                sh = df.groupby(["room_type", "superhost"])["price"].median().reset_index()
                sh["Host"] = sh["superhost"].map({0: "Other host", 1: "Superhost"})
                fig = px.bar(sh, x="room_type", y="price", color="Host", barmode="group",
                             color_discrete_map={"Other host": GRAY, "Superhost": CORAL},
                             labels={"room_type": "", "price": "Median nightly price ($)"})
                fig.update_layout(margin=dict(t=10))
                chart(fig, 320)
            with c2:
                bins = pd.cut(df["availability_365"], [-1, 90, 180, 270, 365],
                              labels=["0–90", "91–180", "181–270", "271–365"])
                av = df.groupby(bins, observed=True)["price"].agg(Median="median", n="size").reset_index()
                fig = go.Figure(go.Bar(x=av["availability_365"].astype(str), y=av["Median"], marker_color=NAVY,
                                       text=av["Median"].map(money), textposition="outside", customdata=av["n"],
                                       hovertemplate="%{x} days<br>Median %{y:$,.0f}<br>%{customdata:,} listings<extra></extra>"))
                fig.update_layout(xaxis_title="Days available in the next year", yaxis_title="Median nightly price ($)",
                                  margin=dict(t=10), yaxis=dict(range=[0, av["Median"].max() * 1.2]))
                chart(fig, 320)
            shm = df.groupby("superhost")["price"].median()
            note(
                f"Superhosts ask a higher median price ({money(shm.get(1, np.nan))} versus "
                f"{money(shm.get(0, np.nan))}) within most room types, and the adjusted premium is about "
                f"{pct('superhost'):.0f}%. Availability has only a small effect: 30 more open days goes with about "
                f"0.4% lower prices, and open days can mean either low demand or a host who blocks fewer dates."
            )

    else:
        rd = df.dropna(subset=["review_scores_rating"])
        if len(rd) < 40:
            st.info("Too few rated listings match the sidebar filters, so these charts use all rated listings.")
            rd = df_all.dropna(subset=["review_scores_rating"])
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            card_header("⭐", "How guests rate listings")
            fig = px.histogram(rd, x="review_scores_rating", nbins=40, color_discrete_sequence=[GOLD],
                               labels={"review_scores_rating": "Overall rating (out of 5)"})
            fig.update_layout(yaxis_title="Listings", margin=dict(t=10), bargap=0.05)
            chart(fig, 330)
            note(
                f"Ratings pile up at the top: the median is {rd['review_scores_rating'].median():.2f} and "
                f"{100 * (rd['review_scores_rating'] >= 4.8).mean():.0f}% of rated listings score 4.8 or higher. "
                f"With so little spread, ratings are hard to explain, and our ratings regression accounts for only "
                f"about {100 * meta['rating_model']['r2']:.0f}% of their variation."
            )
        with c2, st.container(border=True):
            card_header("💸", "Do pricier listings rate better?")
            q = min(10, max(2, len(rd) // 50))
            rd = rd.assign(decile=pd.qcut(rd["price"].rank(method="first"), q, labels=False) + 1)
            rr = rd.groupby("decile").agg(r=("review_scores_rating", "mean"), lo=("price", "min"),
                                          hi=("price", "max"))
            fig = go.Figure(go.Scatter(
                x=rr.index, y=rr["r"], mode="lines+markers", line=dict(color=GOLD, width=3),
                marker=dict(size=10, color=GOLD, line=dict(color="white", width=2)),
                customdata=np.stack([rr["lo"], rr["hi"]], axis=1),
                hovertemplate="Price group %{x}: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}<br>"
                              "Avg rating %{y:.2f}<extra></extra>"))
            fig.update_layout(xaxis_title=f"Price group (1 = cheapest, {q} = priciest)",
                              yaxis_title="Average rating", margin=dict(t=10), xaxis=dict(dtick=1))
            chart(fig, 330)
            note(
                f"Average rating drifts up from {rr['r'].iloc[0]:.2f} in the cheapest group to {rr['r'].iloc[-1]:.2f} "
                f"in the priciest, but the rank correlation across all listings is only "
                f"{meta['rating_model']['spearman']:.2f}. In the adjusted model, doubling the price goes with just "
                f"{meta['rating_model']['log_price_coef'] * np.log(2):.2f} extra stars, while being a Superhost adds "
                f"{meta['rating_model']['superhost_coef']:.2f}."
            )

# ================================================================ TAB 3
with tab3:
    tbl = pd.DataFrame(metrics).T.astype(float)
    tbl.insert(0, "In-sample R² (notebook)",
               [f'{meta["insample_ols"][m]["R2"]:.3f}' if m in meta["insample_ols"] else "–" for m in tbl.index])
    tbl = tbl.rename(columns={"R2 (log price)": "Test R²", "RMSE (log price)": "Test RMSE (log)",
                              "MAE ($)": "Test MAE ($)", "Median abs. % error": "Median abs. error (%)"})

    with st.container(border=True):
        card_header("🏆", "Seven models, one held-out test set",
                    f"Every model predicts log nightly price. Fit on {meta['n_train']:,} listings (80%), scored on "
                    f"{meta['n_test']:,} listings it never saw (20%). The four OLS models are the nested regressions "
                    f"from our notebook; Ridge, Random Forest and Gradient Boosting were tuned by cross-validation on "
                    f"the training set only.")
        st.write("")
        metric_opts = {"Test R²": True, "Median abs. error (%)": False, "Test MAE ($)": False, "Test RMSE (log)": False}
        mcol = st.segmented_control("Compare on", list(metric_opts), default="Test R²", key="metric") or "Test R²"
        higher = metric_opts[mcol]
        vals = tbl[mcol]
        fmt = {"Test R²": "{:.3f}", "Median abs. error (%)": "{:.1f}%", "Test MAE ($)": "${:,.0f}",
               "Test RMSE (log)": "{:.3f}"}[mcol]
        winner = vals.idxmax() if higher else vals.idxmin()
        colors = [CORAL if m == winner else ("#A7B0D6" if m.startswith("OLS") else NAVY) for m in order]
        fig = go.Figure(go.Bar(x=[o.replace(": ", "<br>") for o in order], y=vals[order], marker_color=colors,
                               text=[fmt.format(v) for v in vals[order]], textposition="outside",
                               hovertemplate="%{x}<br>" + mcol + ": %{text}<extra></extra>"))
        fig.update_layout(yaxis_title=mcol + (" (higher is better)" if higher else " (lower is better)"),
                          yaxis=dict(range=[0, vals.max() * 1.18]), margin=dict(t=20))
        chart(fig, 380)
        note(
            f"Light bars are the nested OLS models, dark bars the tuned models, and coral marks the best on "
            f"{mcol}. Each OLS step helps, with neighborhood giving the biggest single jump; Ridge matches the full "
            f"OLS almost exactly, showing the regression was not overfitting. The tree models win because they catch "
            f"interactions, such as a pool being worth more in the hills than in a studio."
        )
        st.dataframe(
            tbl[["Features", "In-sample R² (notebook)", "Test R²", "Test RMSE (log)", "Test MAE ($)",
                 "Median abs. error ($)", "Median abs. error (%)"]]
            .style.format({"Features": "{:.0f}", "Test R²": "{:.3f}",
                           "Test RMSE (log)": "{:.3f}", "Test MAE ($)": "${:,.0f}",
                           "Median abs. error ($)": "${:,.0f}", "Median abs. error (%)": "{:.1f}%"}, na_rep="–")
            .highlight_max(subset=["Test R²"], color="#FDE6E7")
            .highlight_min(subset=["Test RMSE (log)", "Test MAE ($)", "Median abs. error ($)",
                                   "Median abs. error (%)"], color="#FDE6E7"),
            width="stretch")
        st.caption("In-sample R² comes from the notebook's full-sample regressions and applies only to the OLS models.")

    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1, st.container(border=True):
        card_header("🎯", "Predicted vs. actual price")
        cc1, cc2 = st.columns([2, 1])
        pick = cc1.selectbox("Model", order, index=order.index(best_model), key="pva")
        by_room = cc2.toggle("Color by room type", value=True)
        pv = pd.DataFrame({"Actual": np.exp(preds["actual_log"]), "Predicted": np.exp(preds[pick]),
                           "Room type": preds["room_type"]})
        ticks = [10, 30, 100, 300, 1000, 3000, 10000]
        fig = px.scatter(pv, x="Actual", y="Predicted", log_x=True, log_y=True, opacity=0.45,
                         color="Room type" if by_room else None, color_discrete_map=ROOM_COLORS,
                         color_discrete_sequence=[NAVY])
        lo, hi = 10, float(max(pv["Actual"].max(), pv["Predicted"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(color=INK, dash="dash", width=1.5)))
        fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
        fig.update_layout(xaxis=dict(tickvals=ticks, title="Actual price ($, log scale)"),
                          yaxis=dict(tickvals=ticks, title="Predicted price ($, log scale)"), margin=dict(t=10))
        chart(fig, 470)
        m = metrics[pick]
        note(
            f"Each dot is a test listing; dots on the dashed line are perfect predictions. {pick} reaches R² "
            f"{m['R2 (log price)']:.3f} and misses the typical listing by {m['Median abs. % error']:.0f}%, with the "
            f"widest misses among a handful of luxury homes that no model prices well."
        )
    with c2, st.container(border=True):
        card_header("📏", "Error by room type")
        err = pd.DataFrame({"Room type": preds["room_type"],
                            "Abs % error": 100 * np.abs(np.exp(preds[pick]) / np.exp(preds["actual_log"]) - 1)})
        er = err.groupby("Room type")["Abs % error"].median().sort_values()
        fig = go.Figure(go.Bar(x=er.values, y=er.index, orientation="h",
                               marker_color=[ROOM_COLORS.get(r, NAVY) for r in er.index],
                               text=[f"{v:.0f}%" for v in er.values], textposition="outside"))
        fig.update_layout(xaxis_title="Median absolute % error", margin=dict(t=10),
                          xaxis=dict(range=[0, er.max() * 1.3]))
        chart(fig, 250)
        note(f"Where {pick} struggles: errors are smallest for the most common listing type and larger for "
             f"rare room types with few training examples.")

        card_header("⚙️", "Best hyperparameters")
        hp_model = st.pills("Model", ["Gradient Boosting", "Random Forest", "Ridge"], default="Gradient Boosting",
                            key="hp") or "Gradient Boosting"
        params = meta["best_params"][hp_model]
        chips = "".join(f'<span class="chip">{k} = {v}</span>' for k, v in params.items() if k != "CV RMSE (log)")
        st.markdown(chips + f'<br><span class="chip win">CV RMSE (log) = {params["CV RMSE (log)"]}</span>',
                    unsafe_allow_html=True)
        st.write("")
        note("Ridge: 5-fold grid search over alpha. Random Forest (12 candidates) and Gradient Boosting "
             "(20 candidates): 5-fold randomized search. All scored by RMSE on log price. The OLS models have no "
             "hyperparameters.")

# ================================================================ TAB 4
with tab4:
    left, right = st.columns([1, 1.35], gap="large")
    with left, st.container(border=True):
        card_header("🏠", "Describe a listing", "Anything left on <b>Average</b> stays at the LA average.")
        st.write("")
        model_name = st.selectbox("Model", order, index=order.index(best_model))
        room = st.pills("Room type", meta["room_types"], default=d["room_type"], key="room") or d["room_type"]
        hood = st.selectbox("Neighborhood", meta["neighbourhoods"],
                            index=meta["neighbourhoods"].index(d["neighbourhood_cleansed"]))
        beds = st.slider("Bedrooms", 0, 8, int(round(d["bedrooms"])))
        baths = st.slider("Bathrooms", 0.0, 6.0, float(round(d["bathrooms_num"] * 2) / 2), step=0.5)
        guests = st.slider("Guests", 1, 16, int(round(d["accommodates"])))

        def tri(label, key):
            opt = st.segmented_control(label, ["Average", "Yes", "No"], default="Average", key=f"r_{key}",
                                       help=f"Average = {100 * d[key]:.0f}% of listings have it") or "Average"
            return d[key] if opt == "Average" else (1.0 if opt == "Yes" else 0.0)

        sh = tri("Superhost", "superhost")
        pool = tri("Pool / spa", "pool_spa")
        ac = tri("Air conditioning", "air_conditioning")

    base = {k: d[k] for k in ["availability_365"] + AMENITIES}
    base.update({"room_type": room, "neighbourhood_cleansed": hood, "bedrooms": beds, "bathrooms_num": baths,
                 "accommodates": guests, "superhost": sh, "pool_spa": pool, "air_conditioning": ac})

    def predict(name, rows):
        mdl = models[name]
        X = pd.DataFrame(rows)
        return np.exp(mdl["pipeline"].predict(X[mdl["features"]]))

    with right:
        price = float(predict(model_name, [base])[0])
        err = metrics[model_name]["Median abs. % error"]
        hood_med = df_all.loc[df_all["neighbourhood_cleansed"] == hood, "price"].median()
        with st.container(border=True):
            c1, c2 = st.columns([1.3, 1])
            with c1:
                st.markdown(f'<div class="note">Predicted nightly price · {model_name}</div>'
                            f'<div class="big-price">{money(price)}</div>'
                            f'<div class="range">Likely range {money(price * (1 - err / 100))} – '
                            f'{money(price * (1 + err / 100))} (±{err:.0f}% typical error)</div>',
                            unsafe_allow_html=True)
            with c2:
                st.metric(f"Actual median in {hood}", money(hood_med),
                          f"prediction is {100 * (price / hood_med - 1):+.0f}% vs this", delta_color="off")
            unused = [LABELS[f] for f in ["neighbourhood_cleansed", "superhost", "pool_spa", "air_conditioning"]
                      if f not in models[model_name]["features"]]
            if unused:
                st.info(f"{model_name} does not use {', '.join(unused)}, so those inputs have no effect here.")

        with st.container(border=True):
            card_header("🧪", "What if?", "Vary one feature and keep the rest as set on the left.")
            wf = st.segmented_control("Feature to vary", ["Bedrooms", "Guests", "Bathrooms", "Neighborhood"],
                                      default="Bedrooms", key="wf") or "Bedrooms"
            show_all = st.toggle("Show every model", value=False)
            mlist = order if show_all else [model_name]
            if wf == "Neighborhood":
                hoods = df_all["neighbourhood_cleansed"].value_counts().head(15).index.tolist()
                if hood not in hoods:
                    hoods.append(hood)
                rows = [{**base, "neighbourhood_cleansed": h} for h in hoods]
                res = pd.DataFrame({"Neighborhood": hoods, "Price": predict(model_name, rows)}).sort_values("Price")
                fig = go.Figure(go.Bar(x=res["Price"], y=res["Neighborhood"], orientation="h",
                                       marker_color=[CORAL if h == hood else "#A7B0D6" for h in res["Neighborhood"]],
                                       text=res["Price"].map(money), textposition="outside"))
                fig.update_layout(xaxis_title=f"Predicted price ({model_name})", margin=dict(t=10),
                                  xaxis=dict(range=[0, res["Price"].max() * 1.2]))
                chart(fig, 460)
                note("The same listing priced in the 15 busiest neighborhoods (coral = your pick). OLS 1–3 ignore "
                     "neighborhood, so they show flat bars.")
            else:
                col, grid = {"Bedrooms": ("bedrooms", list(range(0, 9))),
                             "Guests": ("accommodates", list(range(1, 17))),
                             "Bathrooms": ("bathrooms_num", list(np.arange(0.5, 6.5, 0.5)))}[wf]
                rows = [{**base, col: v} for v in grid]
                fig = go.Figure()
                for mname in mlist:
                    y = predict(mname, rows)
                    is_sel = mname == model_name
                    fig.add_trace(go.Scatter(x=grid, y=y, mode="lines+markers", name=mname,
                                             line=dict(width=3.5 if is_sel else 1.5,
                                                       color=CORAL if is_sel else None),
                                             marker=dict(size=7 if is_sel else 4),
                                             hovertemplate=f"{mname}<br>{wf}: %{{x}}<br>%{{y:$,.0f}}<extra></extra>"))
                cur = base[col]
                fig.add_trace(go.Scatter(x=[cur], y=[price], mode="markers", name="Your listing",
                                         marker=dict(size=16, color=INK, symbol="star", line=dict(color="white", width=2))))
                fig.update_layout(xaxis_title=wf, yaxis_title="Predicted nightly price ($)", margin=dict(t=10),
                                  showlegend=True)
                chart(fig, 400)
                note(f"How the predicted price moves as {wf.lower()} changes. Linear models give smooth "
                     f"exponential curves, while tree models step and level off where the data gets thin.")

        with st.expander("Features held at their averages"):
            held = {LABELS[k]: (f"{v:.0f} days" if k == "availability_365" else f"{100 * v:.0f}% of listings")
                    for k, v in d.items() if k in ["availability_365", "wifi", "parking", "kitchen", "washer_dryer"]}
            st.dataframe(pd.Series(held, name="Value used"), width="stretch")

st.markdown('<div class="note" style="text-align:center;margin-top:30px">Associations from observational listing '
            'data, not causal effects. Models are loaded from saved files; nothing is retrained in this app.</div>',
            unsafe_allow_html=True)
