"""Federated churn demo STARTER — teaching skeleton, not solution.

Run:  churnenv Scripts activate, then `streamlit run frontend/federated_demo.py`
V1 = pre-recorded: you run `flwr run` in terminal, paste numbers into fl_rounds_sample.csv.
Live `flwr run` button is deliberately NOT here (see docs/fed-guide.md Lab 3 + Next Steps).

TODOs are yours to complete. If you can explain each TODO, you can defend it.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Federated Churn Demo", layout="wide")

st.title("Privacy-Based Churn — Federated Demo (starter)")
st.caption("Learn → Run (pre-recorded) → Compare → Predict. See docs/fed-guide.md.")

tab_learn, tab_run, tab_compare, tab_predict = st.tabs(
    ["1. Learn", "2. Run", "3. Compare", "4. Predict"]
)

with tab_learn:
    st.subheader("One-round story (say it aloud)")
    st.markdown(
        """
        1. Server sends global trees (bytes in `Message.content["arrays"]["0"]`).
        2. Each branch boosts new trees on **local rows only** (`ClientApp.train`).
        3. Server concatenates trees (**FedXgbBagging**, not averaging).
        4. Repeat. Rows never leave branches; only trees + `auc` / `num-examples` move.

        **TODO-LEARN:** in your words (2-3 bullets), why can't we FedAvg trees?
        Hint: open `quickstart-xgboost/quickstart_xgboost/client_app.py:20-30`.
        """
    )

with tab_run:
    st.subheader("Federated rounds (pre-recorded CSV)")

    @st.cache_data
    def load_rounds(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    csv_path = Path(__file__).parent / "fl_rounds_sample.csv"
    df = load_rounds(str(csv_path))

    st.dataframe(df, width="stretch")
    st.line_chart(df.set_index("round")[["auc"]])

    st.info(
        "TODO-RUN: run Lab 2 Telco federated run (`flwr run . --stream` in quickstart-xgboost), "
        "create fl_rounds.csv with your real round→AUC, change line 45 to load it. "
        "Add st.metric cards for final AUC and total examples."
    )

with tab_compare:
    st.subheader("Centralized vs federated")

    central_auc = st.slider("Centralized AUC (from your notebook, Lab 0)", 0.5, 1.0, 0.82, 0.01)
    # TODO-COMPARE: replace 0.81 with your federated final AUC from Lab 2
    fed_auc = float(df["auc"].iloc[-1]) if len(df) else 0.81

    comp = pd.DataFrame({"model": ["centralized", "federated"], "auc": [central_auc, fed_auc]})
    st.bar_chart(comp.set_index("model"))

    st.info("TODO-COMPARE: write one sentence — why is federated slightly lower/higher? Think Non-IID.")

with tab_predict:
    st.subheader("Predict (simplified, federated model)")

    @st.cache_resource
    def load_federated_model(path: str):
        import xgboost as xgb

        p = Path(path)
        if not p.exists():
            return None
        bst = xgb.Booster()
        bst.load_model(str(p))
        return bst

    # Lab 2 produces quickstart-xgboost/final_model.json when save-model=true
    model = load_federated_model("quickstart-xgboost/final_model.json")
    if model is None:
        st.warning(
            "No `quickstart-xgboost/final_model.json` yet — finish Lab 2 first. "
            "This tab will light up after your first Telco federated run."
        )
    else:
        st.success(f"Loaded federated model with {model.num_boosted_rounds()} trees.")

    with st.form("mini_churn"):
        tenure = st.slider("Tenure in Months", 0, 72, 12)
        monthly = st.slider("Monthly Charge", 0.0, 200.0, 70.0)
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        st.form_submit_button("Predict (TODO)")

    st.info(
        "TODO-PREDICT: map these 3 fields + 5 more of your choice to the 52 "
        "`FEATURE_NAMES` order in `API/app.py:33-86` (hint: `reindex(columns=FEATURE_NAMES)`), "
        "build 1-row DMatrix/DataFrame, call `model.predict`. Keep full 52-field form in `streamlit_app.py`."
    )
