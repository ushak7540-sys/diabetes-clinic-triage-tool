import numpy as np, pandas as pd, joblib, shap, streamlit as st

st.set_page_config(page_title="Diabetes Clinic Triage Tool", page_icon="🩸")
st.markdown("""
<style>
.stApp {background: linear-gradient(160deg,#e0f2fe 0%,#bfdbfe 45%,#eff6ff 100%);}
h1 {color:#1d4ed8; border-bottom:4px solid #dc2626; padding-bottom:6px;}
</style>""", unsafe_allow_html=True)

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

@st.cache_resource
def load():
    art = joblib.load("triage_model.pkl")
    model = art["model"]
    bg = pd.read_csv("train_background.csv")
    explainer = shap.Explainer(model[-1], bg)
    return model, art["high_cut"], art["med_cut"], art["features"], explainer

model, high_cut, med_cut, FEATURES, explainer = load()
prep = model[:-1]

def risk_label(p):
    return "High" if p >= high_cut else "Medium" if p >= med_cut else "Low"

def triage(vals):
    row = pd.DataFrame([vals], columns=FEATURES, dtype=float)
    row[ZERO_COLS] = row[ZERO_COLS].replace(0, np.nan)
    n_missing = int(row.isnull().sum(axis=1).iloc[0])
    note = (f"<p style='color:#92400e;'>⚠️ {n_missing} field(s) missing, "
            f"treat this score with caution</p>") if n_missing else ""
    p = model.predict_proba(row)[0, 1]
    label = risk_label(p)
    icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[label]
    action = {"High": "Send for full blood test today",
              "Medium": "Test if lab capacity allows",
              "Low": "Routine follow-up"}[label]
    x = pd.DataFrame(prep.transform(row), columns=FEATURES)
    sv = explainer(x)
    v = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    top = pd.Series(v, index=FEATURES).sort_values(ascending=False).head(3)
    reasons = [f"<li><b>{f}</b> = {row[f].iloc[0]:.1f} raises risk</li>"
               for f, s in top.items() if s > 0 and not np.isnan(row[f].iloc[0])]
    colors = {"High": ("#fee2e2", "#dc2626"), "Medium": ("#fef9c3", "#ca8a04"),
              "Low": ("#dcfce7", "#16a34a")}
    bg, border = colors[label]
    return f"""
    <div style="background:{bg}; border-left:8px solid {border}; color:#111;
                padding:18px; border-radius:14px; font-size:16px;">
      <h2 style="margin:0; color:{border};">{icon} {label} risk: {p:.0%}</h2>
      <p><b>Action:</b> {action}</p>
      <b>Why flagged:</b><ul>{"".join(reasons)}</ul>{note}
    </div>"""

def rank(df, cap):
    d = df[FEATURES].copy()
    d[ZERO_COLS] = d[ZERO_COLS].replace(0, np.nan)
    d["Risk %"] = (model.predict_proba(d)[:, 1] * 100).round(1)
    d = d.sort_values("Risk %", ascending=False).reset_index(drop=True)
    n = int(np.ceil(len(d) * cap / 100))
    d.insert(0, "Rank", d.index + 1)
    d["Decision"] = ["🔴 TEST TODAY" if i < n else "⚪ Not today" for i in range(len(d))]
    return d, n

st.title("🩸 Diabetes Clinic Triage Tool")
st.caption("🔵 Quick diabetes risk screening for rural clinics. Screening aid only, not a diagnosis.")

tab1, tab2 = st.tabs(["🧍 Single Patient", "📋 Daily Queue"])

with tab1:
    st.write("Enter 0 for any value that is not available.")
    c1, c2 = st.columns(2)
    preg = c1.number_input("🤰 Pregnancies", 0.0, 20.0, 0.0, 1.0)
    glu = c1.number_input("🍬 Glucose (mg/dL)", 0.0, 300.0, 0.0, 1.0)
    bp = c1.number_input("❤️ Blood Pressure (mm Hg)", 0.0, 200.0, 0.0, 1.0)
    skin = c1.number_input("📏 Skin Thickness (optional)", 0.0, 100.0, 0.0, 1.0)
    ins = c2.number_input("💉 Insulin (optional)", 0.0, 900.0, 0.0, 1.0)
    bmi = c2.number_input("⚖️ BMI", 0.0, 70.0, 0.0, 0.1)
    dpf = c2.number_input("🧬 Family History Score", 0.0, 2.5, 0.0, 0.01)
    age = c2.number_input("🎂 Age", 0.0, 120.0, 0.0, 1.0)
    if st.button("Check risk", type="primary"):
        st.markdown(triage([preg, glu, bp, skin, ins, bmi, dpf, age]),
                    unsafe_allow_html=True)

with tab2:
    file = st.file_uploader("Upload today's walk-in CSV", type="csv")
    cap = st.slider("🧪 Lab capacity (% of patients testable today)", 10, 60, 30, 5)
    if st.button("Rank patients", type="primary"):
        if file is None:
            st.warning("Please upload a CSV first")
        else:
            d, n = rank(pd.read_csv(file), cap)
            st.markdown(f"### {n} of {len(d)} patients selected for blood test today (top {cap}%)")
            st.dataframe(d)
            st.download_button("Download ranked list", d.to_csv(index=False), "queue_result.csv")
