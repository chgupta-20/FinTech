"""
Interactive Streamlit Dashboard for Smart Wealth (Inter-IIT Tech Meet 15.0).
Launch command:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.engine import recommend_for_investor

st.set_page_config(page_title="Smart Wealth | Goal-Based Investment Engine", layout="wide")

@st.cache_data
def load_datasets():
    products = pd.read_csv("data/investment_products.csv")
    profiles = pd.read_csv("data/investor_profiles.csv")
    returns = pd.read_csv("data/synthetic_monthly_returns.csv")
    return products, profiles, returns

products_df, profiles_df, returns_df = load_datasets()

st.title("Smart Wealth: Goal-Based Investment Strategy Engine")
st.caption(
    "Educational Planning Prototype | Grounded in Mental Accounting (Das, Markowitz, Scheid & Statman, 2010), "
    "Behavioral Portfolio Pyramids (Shefrin & Statman, 2000), and Dynamic GBWM (Das et al., 2020)."
)

# Sidebar: Persona Selection & Live What-If Input Controls (Section 6)
st.sidebar.header("1. Select Investor Persona")
persona_options = [
    f"{row['investor_id']} - {row['goal_label']} ({row['horizon_months']}m)"
    for _, row in profiles_df.iterrows()
]
selected_label = st.sidebar.selectbox("Choose Persona", persona_options)
selected_id = selected_label.split(" - ")[0]
base_row = profiles_df[profiles_df["investor_id"] == selected_id].iloc[0].to_dict()

st.sidebar.header("2. Interactive What-If Adjustments")
st.sidebar.markdown("Modify inputs below to see live re-optimization and updated projections:")

goal_amount = st.sidebar.number_input("Goal Amount (INR)", min_value=10000.0, value=float(base_row["goal_amount_inr"]), step=50000.0)
horizon_months = st.sidebar.slider("Goal Horizon (Months)", min_value=6, max_value=300, value=int(base_row["horizon_months"]), step=3)
existing_inv = st.sidebar.number_input("Existing Investment (INR)", min_value=0.0, value=float(base_row["existing_investment_inr"]), step=25000.0)
monthly_sip = st.sidebar.number_input("Monthly SIP (INR)", min_value=0.0, value=float(base_row["monthly_sip_inr"]), step=1000.0)
risk_tol = st.sidebar.slider("Risk Tolerance Score (1-5)", min_value=1.0, max_value=5.0, value=float(base_row["risk_tolerance_score"]), step=0.5)
emerg_months = st.sidebar.number_input("Emergency Fund Coverage (Months)", min_value=0.0, max_value=60.0, value=float(base_row["emergency_fund_months"]), step=0.5)
liq_need = st.sidebar.number_input("Liquidity Need (INR)", min_value=0.0, value=float(base_row["liquidity_need_inr"]), step=10000.0)

pref_default = "none" if pd.isna(base_row["preference"]) else str(base_row["preference"])
pref_list = ["none", "high_growth", "capital_protection", "low_cost", "avoid_equity", "tax_saving", "no_lock_in"]
preference = st.sidebar.selectbox("Optional Preference", pref_list, index=pref_list.index(pref_default) if pref_default in pref_list else 0)

live_profile = {
    "investor_id": selected_id,
    "goal_label": base_row["goal_label"],
    "goal_amount_inr": goal_amount,
    "horizon_months": horizon_months,
    "existing_investment_inr": existing_inv,
    "monthly_sip_inr": monthly_sip,
    "risk_tolerance_score": risk_tol,
    "emergency_fund_months": emerg_months,
    "liquidity_need_inr": liq_need,
    "experience": base_row["experience"],
    "preference": preference,
}

result = recommend_for_investor(live_profile, products_df, returns_df, include_what_if=True)

if not result["is_valid"]:
    st.error("Invalid Investor Profile Inputs:")
    for err in result["validation_errors"]:
        st.write(f"- {err}")
    st.info(result["fallback_portfolio"]["message"])
    st.stop()

opt = result["optimization"]
proj = result["projections"]
exp = result["explanation"]

# Prominent Warnings Banner (Section 4 Guardrail 5 & Section 6)
if exp["active_warnings"]:
    st.subheader("Active Guardrail & Suitability Warnings")
    for w_msg in exp["active_warnings"]:
        st.warning(w_msg)

if opt["emergency_buffer_plan"]["needs_emergency_buffer"]:
    st.info(f"**BPT Safety Layer Recommendation:** {opt['emergency_buffer_plan']['buffer_explanation']}")

# Top KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Weighted Risk Score", f"{opt['weighted_risk_score']:.2f} / 5.0", f"Max Allowed: {opt['max_allowed_risk_score']:.2f}")
col2.metric("Assumed Portfolio Return", f"{opt['weighted_assumed_return']*100:.2f}% p.a.", f"Net of Fees: {opt['weighted_net_return']*100:.2f}%")
col3.metric("Weighted Expense Ratio", f"{opt['weighted_expense_ratio']*100:.2f}% p.a.", f"INR {opt['estimated_annual_cost_inr']:,.0f}/yr cost")
col4.metric("Base Projected Value", f"INR {proj['fv_base_inr']:,.0f}", f"Target: INR {proj['goal_amount_inr']:,.0f}")
col5.metric("Estimated Surplus / Shortfall", f"INR {proj['surplus_shortfall_base_inr']:+,.0f}", f"Success Prob: {proj['prob_reach_goal']*100:.1f}%")

st.divider()

# Product-Level Allocation & Asset-Class Breakdown
left_col, right_col = st.columns([3, 2])
with left_col:
    st.subheader("Recommended Product-Level Allocation")
    alloc_table = pd.DataFrame({
        "Product ID": products_df["product_id"],
        "Product Name": products_df["product_name"],
        "Asset Class": products_df["asset_class"],
        "Risk Score": products_df["risk_score"],
        "Weight (%)": np.round(opt["weights"] * 100, 2),
        "Initial Allocation (INR)": opt["allocated_existing_inr"],
        "Monthly SIP (INR)": opt["allocated_monthly_sip_inr"],
        "Assumed Return (%)": np.round(products_df["assumed_annual_return"] * 100, 2),
        "Expense Ratio (%)": np.round(products_df["annual_expense_ratio"] * 100, 2),
        "Lock-in (m)": products_df["lock_in_months"],
    })
    st.dataframe(alloc_table, use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Asset Class & Risk Bucket Breakdown")
    ac_df = pd.DataFrame({
        "Asset Class": [k.capitalize() for k in opt["asset_class_breakdown"].keys()],
        "Allocation (%)": [round(v * 100, 2) for v in opt["asset_class_breakdown"].values()],
    })
    st.bar_chart(ac_df.set_index("Asset Class"))
    rb = opt["risk_bucket_breakdown"]
    st.markdown(
        f"- **Horizon Bucket:** {rb['horizon_bucket']}\n"
        f"- **Low Risk (Score 1-2):** `{rb['risk_1_2']*100:.1f}%` (Guardrail Min: `{rb['min_r12_limit']*100:.0f}%`)\n"
        f"- **Moderate Risk (Score 3):** `{rb['risk_3']*100:.1f}%`\n"
        f"- **High Risk (Score 4-5):** `{rb['risk_4_5']*100:.1f}%` (Guardrail Max: `{rb['max_r45_limit']*100:.0f}%`)\n"
        f"- **Liquidity Status:** {opt['liquidity_status']}"
    )

st.divider()

# Goal Projection Range & Trajectory Chart
st.subheader("Goal Feasibility & Deterministic Projection Range (Educational Estimate)")
pcol1, pcol2 = st.columns([2, 1])
with pcol1:
    traj_df = pd.DataFrame({
        "Month": proj["trajectory"]["months"],
        "Total Principal": proj["trajectory"]["principal"],
        "Conservative (r_p - 3%)": proj["trajectory"]["fv_low"],
        "Base Case (r_p)": proj["trajectory"]["fv_base"],
        "Optimistic (r_p + 3%)": proj["trajectory"]["fv_high"],
    }).set_index("Month")
    st.line_chart(traj_df)

with pcol2:
    st.markdown("**Scenario Comparison vs Target Goal**")
    scen_table = pd.DataFrame([
        {"Scenario": f"Conservative ({proj['r_low']*100:.2f}%)", "Projected FV (INR)": f"{proj['fv_low_inr']:,.0f}", "Surplus / Shortfall (INR)": f"{proj['surplus_shortfall_low_inr']:+,.0f}"},
        {"Scenario": f"Base Case ({proj['r_base']*100:.2f}%)", "Projected FV (INR)": f"{proj['fv_base_inr']:,.0f}", "Surplus / Shortfall (INR)": f"{proj['surplus_shortfall_base_inr']:+,.0f}"},
        {"Scenario": f"Net of Fees ({proj['r_net']*100:.2f}%)", "Projected FV (INR)": f"{proj['fv_net_of_fees_inr']:,.0f}", "Surplus / Shortfall (INR)": f"{proj['surplus_shortfall_net_inr']:+,.0f}"},
        {"Scenario": f"Optimistic ({proj['r_high']*100:.2f}%)", "Projected FV (INR)": f"{proj['fv_high_inr']:,.0f}", "Surplus / Shortfall (INR)": f"{proj['surplus_shortfall_high_inr']:+,.0f}"},
    ])
    st.table(scen_table)
    st.markdown(f"**Actionable Recourse Plan:** {proj['recourse_recommendation']}")

st.divider()

# Explainability Section (Objective 5)
st.subheader("Explainability, Downside Trade-Offs & Excluded Alternatives")
st.markdown("**Why This Portfolio Fits Your Profile:**")
for reason in exp["profile_reasons"]:
    st.markdown(f"- {reason}")

st.markdown(f"**Main Portfolio Downside / Trade-Off:** {exp['main_downside_tradeoff']}")

ecol1, ecol2 = st.columns(2)
with ecol1:
    st.markdown("**Included Products & Constraints Served:**")
    st.dataframe(pd.DataFrame(exp["included_products_rationale"])[["product_id", "product_name", "weight_pct", "rationale"]], use_container_width=True, hide_index=True)
with ecol2:
    st.markdown("**Why Plausible Alternatives Were Limited or Excluded:**")
    st.dataframe(pd.DataFrame(exp["excluded_alternatives"]), use_container_width=True, hide_index=True)

st.divider()

# Automated What-If Stress Table (Objective 6)
st.subheader("Automated Multi-Scenario What-If Stress Analysis")
st.dataframe(pd.DataFrame(result["what_if_scenarios"]), use_container_width=True, hide_index=True)
st.caption(exp["educational_disclaimer"])