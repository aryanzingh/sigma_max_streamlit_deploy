from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Sigma Max Predictor", page_icon="🔧")

@st.cache_resource
def load():
    return joblib.load(Path(__file__).parent / "sigma_max_bundle.joblib")

B = load()
model, cols, base = B["model"], B["input_columns"], B["base"]

def fe(X):
    Z = X.copy()
    for c in base:
        Z[c + "_sq"] = X[c] ** 2
        if (X[c] > 0).all():
            Z[c + "_log"] = np.log(X[c])
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            Z[f"{a}*{b}"] = X[a] * X[b]
            if (X[b] != 0).all(): Z[f"{a}/{b}"] = X[a] / X[b]
            if (X[a] != 0).all(): Z[f"{b}/{a}"] = X[b] / X[a]
    return Z.replace([np.inf, -np.inf], np.nan).fillna(0)

st.title("🔧 Sigma Max (Maximum Stress) Predictor")
st.caption("Enter the plate parameters to estimate maximum stress.")

vals = {}
left, right = st.columns(2)
for i, c in enumerate(cols):
    lo, hi, med = B["stats"][c]
    with (left if i % 2 == 0 else right):
        vals[c] = st.number_input(c, value=med, help=f"Training range: {lo:.4g} to {hi:.4g}")

if st.button("Predict Sigma Max", type="primary"):
    row = pd.DataFrame([vals])[cols].astype(float)
    if B["use_fe"]:
        row = fe(row)
    row = row.reindex(columns=B["model_columns"], fill_value=0)
    pred = float(model.predict(row)[0])
    st.success(f"Predicted Sigma Max: **{pred:,.4f}**")
    out_of_range = [c for c in cols if not (B["stats"][c][0] <= vals[c] <= B["stats"][c][1])]
    if out_of_range:
        st.warning(f"Outside the training range, so the prediction may be unreliable: {', '.join(out_of_range)}")