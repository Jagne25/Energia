# app.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# ZÁKLAD
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
st.set_page_config(page_title="Energia – Dashboard", layout="wide")

st.sidebar.title("⚡ Energia – Dashboard")
st.sidebar.markdown(
    """
    Tento panel zobrazuje:
    - **prehľad metrík** (auto z `results.csv`, `*_results.csv`, aj `*_results.json`),
    - **predpovede** z modelov (LSTM / Baseline / SARIMA / Prophet),
    - voliteľné porovnanie so **skutočnými** hodnotami.
    """
)

# ----------------------------------------------------------------------
# NAČÍTAVANIE & UTILITY
# ----------------------------------------------------------------------
def read_csv_safe(p: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as e:
        st.warning(f"Neviem načítať CSV {p.name}: {e}")
        return None

def read_json_safe(p: Path) -> Optional[Any]:
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Neviem načítať JSON {p.name}: {e}")
        return None

def to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------------------------------------------------
# LEADERBOARD – AUTO-NAČÍTANIE VÝSLEDKOV
# ----------------------------------------------------------------------
def normalize_results_df(df: pd.DataFrame, model_fallback: str) -> pd.DataFrame:
    """Z df vytiahne stĺpce Model/MAE/RMSE/MAPE/Poznámka (ak sa dajú nájsť)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Model","MAE","RMSE","MAPE","Poznámka"])

    lower = {c.lower(): c for c in df.columns}
    model_col = lower.get("model", None)
    mae_col   = lower.get("mae", None)
    rmse_col  = lower.get("rmse", None)
    mape_col  = lower.get("mape", None)
    note_col  = lower.get("note", lower.get("poznámka", None))

    out = pd.DataFrame({
        "Model": df[model_col] if model_col else [model_fallback]*len(df),
        "MAE": df[mae_col] if mae_col else np.nan,
        "RMSE": df[rmse_col] if rmse_col else np.nan,
        "MAPE": df[mape_col] if mape_col else np.nan,
        "Poznámka": df[note_col] if note_col else None
    })
    out = coerce_numeric(out, ["MAE","RMSE","MAPE"])
    # pre istotu len jeden riadok na súbor – ak by ich bolo viac, necháme všetky
    return out

def results_from_json(obj: Any) -> pd.DataFrame:
    """Podpora pre JSON štruktúry:
       - slovník model->metriky  alebo
       - objekt s kľúčmi MAE/RMSE/MAPE (jediný model)
    """
    rows = []
    if isinstance(obj, dict):
        # varianta: {"baseline_daily": {"MAE":..., ...}, "prophet_daily": {...}, ...}
        if all(isinstance(v, dict) for v in obj.values()):
            for model, met in obj.items():
                rows.append({
                    "Model": model,
                    "MAE": met.get("MAE"),
                    "RMSE": met.get("RMSE"),
                    "MAPE": met.get("MAPE"),
                    "Poznámka": met.get("note") or met.get("komentar")
                })
        else:
            # varianta: {"Model":"prophet_daily","MAE":...,"RMSE":...,"MAPE":...}
            rows.append({
                "Model": obj.get("Model","unknown"),
                "MAE": obj.get("MAE"),
                "RMSE": obj.get("RMSE"),
                "MAPE": obj.get("MAPE"),
                "Poznámka": obj.get("note") or obj.get("komentar")
            })
    return coerce_numeric(pd.DataFrame(rows), ["MAE","RMSE","MAPE"])

def load_all_results() -> pd.DataFrame:
    rows = []

    # 1) hlavný results.csv
    base = DATA_DIR / "results.csv"
    if base.exists():
        df = read_csv_safe(base)
        rows.append(normalize_results_df(df, "results.csv"))

    # 2) akékoľvek *_results.csv (okrem results.csv)
    for p in DATA_DIR.glob("*_results.csv"):
        if p.name == "results.csv":
            continue
        df = read_csv_safe(p)
        rows.append(normalize_results_df(df, p.stem.replace("_results","")))

    # 3) akékoľvek *_results.json
    for p in DATA_DIR.glob("*_results.json"):
        obj = read_json_safe(p)
        if obj is None:
            continue
        df = results_from_json(obj)
        if df is not None and not df.empty:
            rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out.drop_duplicates()
        out = coerce_numeric(out, ["MAE","RMSE","MAPE"])
        # default sort: MAPE, potom MAE a RMSE
        sort_cols = [c for c in ["MAPE","MAE","RMSE"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=True, na_position="last")
        return out.reset_index(drop=True)

    return pd.DataFrame(columns=["Model","MAE","RMSE","MAPE","Poznámka"])

# ----------------------------------------------------------------------
# FORECAST – NAČÍTANIE A ÚPRAVA
# ----------------------------------------------------------------------
def tidy_forecast_generic(df: Optional[pd.DataFrame],
                          prefer_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Normalize forecast DF na stĺpce: date, forecast, (lower, upper)."""
    if df is None or df.empty:
        return None

    df = df.copy()
    # nájdi dátum
    for cand in ["date","ds","Datetime","timestamp","Index","index"]:
        if cand in df.columns:
            df["date"] = to_datetime_series(df[cand])
            break
    else:
        # možno je dátum v indexe?
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"date"})
            df["date"] = to_datetime_series(df["date"])
        else:
            return None

    # nájdi forecast
    fcols = ["forecast","yhat", prefer_col, "LSTM_7d_forecast", "SARIMA_7d_forecast",
             "Baseline_7d_forecast", "yhat_lower","yhat_upper"]
    fc = None
    for cand in fcols:
        if cand and cand in df.columns:
            fc = cand
            break
    if fc is None:
        # ak je len jeden nenadpisaný stĺpec okrem 'date', zober ho
        rest = [c for c in df.columns if c != "date"]
        if len(rest) == 1:
            fc = rest[0]
        else:
            return None

    # voliteľné pásma neistoty
    lower = "yhat_lower" if "yhat_lower" in df.columns else None
    upper = "yhat_upper" if "yhat_upper" in df.columns else None

    keep = ["date", fc]
    if lower: keep.append(lower)
    if upper: keep.append(upper)
    out = df[keep].sort_values("date").rename(columns={fc:"forecast"})
    if lower: out = out.rename(columns={lower:"lower"})
    if upper: out = out.rename(columns={upper:"upper"})
    return out

def get_forecast_for(model_choice: str) -> Optional[pd.DataFrame]:
    """Podľa voľby v UI nájde správny súbor."""
    # mapovanie model -> kandidáti CSV
    # podporujem 7-dňové aj „dlhé“ súbory pre Prophet
    candidates = {
        "LSTM – 7 dní": [
            "lstm_7day_forecast.csv", "lstm_forecast.csv"
        ],
        "Baseline – 7 dní": [
            "baseline_7day_forecast.csv", "baseline_forecast.csv"
        ],
        "SARIMA – 7 dní": [
            "sarima_7day_forecast.csv", "sarima_forecast.csv"
        ],
        "Prophet – denné": [
            "prophet_7day_forecast.csv", "prophet_forecast.csv"
        ],
    }
    prefer_col = {
        "LSTM – 7 dní": "LSTM_7d_forecast",
        "Baseline – 7 dní": "Baseline_7d_forecast",
        "SARIMA – 7 dní": "SARIMA_7d_forecast",
        "Prophet – denné": "yhat"
    }.get(model_choice, None)

    for name in candidates.get(model_choice, []):
        p = DATA_DIR / name
        df = read_csv_safe(p)
        out = tidy_forecast_generic(df, prefer_col=prefer_col)
        if out is not None and not out.empty:
            return out
    return None

def apply_tail_days(df_fc: Optional[pd.DataFrame], n_days: int) -> Optional[pd.DataFrame]:
    if df_fc is None or df_fc.empty or n_days <= 0:
        return df_fc
    df_fc = df_fc.sort_values("date")
    # necháme posledných N unikátnych dátumov
    unique_dates = df_fc["date"].dropna().dt.normalize().unique()
    unique_dates = unique_dates[-n_days:]
    return df_fc[df_fc["date"].dt.normalize().isin(unique_dates)]

def load_actuals_default() -> Optional[pd.DataFrame]:
    """Načíta skutočné hodnoty z actuals_daily.csv (ak existuje)."""
    p = DATA_DIR / "actuals_daily.csv"
    if not p.exists():
        return None
    df = read_csv_safe(p)
    if df is None: return None

    # mapovanie stĺpcov
    lc = {c.lower(): c for c in df.columns}
    if "date" in lc:
        df["date"] = to_datetime_series(df[lc["date"]])
    elif "datetime" in lc:
        df["date"] = to_datetime_series(df[lc["datetime"]])
    else:
        return None

    if "actual" in lc:
        df = df.rename(columns={lc["actual"]: "actual"})
    elif "load" in lc:
        df = df.rename(columns={lc["load"]: "actual"})
    else:
        return None

    return df[["date","actual"]].dropna().sort_values("date")

# ----------------------------------------------------------------------
# UI TABS
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Prehľad metrík", "📈 Predpovede"])

# ------------------- TAB 1: LEADERBOARD --------------------------------
with tab1:
    st.subheader("Leaderboard modelov (nižšie = lepšie)")
    df_results = load_all_results()

    if df_results.empty:
        st.info("Nenašiel som žiadne výsledky (results.csv, *_results.csv alebo *_results.json).")
    else:
        metric_choice = st.radio("Zoradiť podľa metriky", ["MAPE","MAE","RMSE"], horizontal=True, index=0)
        if metric_choice in df_results.columns:
            df_results = df_results.sort_values(metric_choice, ascending=True, na_position="last").reset_index(drop=True)

        # Top 3 metriky (ak sú)
        c1, c2, c3 = st.columns(3)
        if "MAPE" in df_results.columns and len(df_results) > 0:
            c1.metric(f"#1 · {df_results.iloc[0]['Model']}", f"{df_results.iloc[0]['MAPE']:.2f}%")
            if len(df_results) > 1:
                c2.metric(f"#2 · {df_results.iloc[1]['Model']}", f"{df_results.iloc[1]['MAPE']:.2f}%")
            if len(df_results) > 2:
                c3.metric(f"#3 · {df_results.iloc[2]['Model']}", f"{df_results.iloc[2]['MAPE']:.2f}%")

        st.dataframe(df_results, use_container_width=True)

        # jednoduchý bar-chart na MAPE
        if "MAPE" in df_results.columns:
            fig = px.bar(df_results, x="Model", y="MAPE", text="MAPE", title="MAPE – čím nižšie, tým lepšie")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="MAPE (%)")
            st.plotly_chart(fig, use_container_width=True)

# ------------------- TAB 2: PREDPOVEDE ----------------------------------
with tab2:
    st.subheader("Predpovede modelov")

    model_choice = st.radio(
        "Vyber model",
        options=["LSTM – 7 dní", "SARIMA – 7 dní", "Baseline – 7 dní", "Prophet – denné"],
        horizontal=True,
        index=0
    )

    days_view = st.number_input("Zobraziť posledných N dní (0 = všetko)", min_value=0, max_value=365, value=7, step=1)

    st.markdown("Ak chceš porovnať so **skutočnými** dátami, vlož CSV so stĺpcami: `date, actual` "
                "(alebo `Datetime, Load`).")

    # upload actuals (voliteľné)
    uploaded = st.file_uploader("Nahrať actuals (voliteľné)", type=["csv"])
    actuals = None
    if uploaded is not None:
        try:
            adf = pd.read_csv(uploaded)
            lc = {c.lower(): c for c in adf.columns}
            if "date" in lc:
                adf["date"] = to_datetime_series(adf[lc["date"]])
            elif "datetime" in lc:
                adf["date"] = to_datetime_series(adf[lc["datetime"]])
            else:
                st.warning("CSV musí obsahovať stĺpec 'date' alebo 'Datetime'.")
                adf = None

            if adf is not None:
                if "actual" in lc:
                    adf = adf.rename(columns={lc["actual"]: "actual"})
                elif "load" in lc:
                    adf = adf.rename(columns={lc["load"]: "actual"})
                else:
                    st.warning("CSV musí obsahovať stĺpec 'actual' alebo 'Load'.")
                    adf = None

            if adf is not None:
                actuals = adf[["date","actual"]].dropna().sort_values("date")
        except Exception as e:
            st.warning(f"Nepodarilo sa načítať nahraté CSV: {e}")
    else:
        # ak nič nenahráš, skús default actuals_daily.csv
        actuals = load_actuals_default()

    # načítaj forecast pre model
    df_fc = get_forecast_for(model_choice)
    if df_fc is None or df_fc.empty:
        st.info("Pre zvolený model som nenašiel forecast CSV v priečinku data/.")
    else:
        df_fc = apply_tail_days(df_fc, days_view)

        title = {
            "LSTM – 7 dní": "LSTM – 7-dňová predpoveď",
            "Baseline – 7 dní": "Baseline – 7-dňová predpoveď",
            "SARIMA – 7 dní": "SARIMA – 7-dňová predpoveď",
            "Prophet – denné": "Prophet – denné predpovede"
        }.get(model_choice, "Predpoveď")

        fig = px.line(df_fc, x="date", y="forecast", title=title)
        fig.update_layout(xaxis_title="", yaxis_title="Load")

        # pásmo neistoty (ak je)
        if "lower" in df_fc.columns and "upper" in df_fc.columns:
            fig.add_traces(px.line(df_fc, x="date", y="lower").data)
            fig.add_traces(px.line(df_fc, x="date", y="upper").data)

        # actuals (ak sú)
        if actuals is not None and not actuals.empty:
            fig.add_traces(px.line(actuals, x="date", y="actual").data)
            fig.update_traces(selector=dict(name="actual"), name="Actual")

        st.plotly_chart(fig, use_container_width=True)

        # malá tabuľka posledných bodov + download
        st.caption("Posledné riadky predikcie")
        st.dataframe(df_fc.tail(7), use_container_width=True)
        st.download_button(
            "Stiahnuť forecast (CSV)",
            data=df_fc.to_csv(index=False).encode("utf-8"),
            file_name=f"{title.replace(' ','_').lower()}.csv",
            mime="text/csv"
        )
        