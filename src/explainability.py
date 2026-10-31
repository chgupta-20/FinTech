"""
Deterministic Explainability & Rationale Engine for Smart Wealth.
Implements Section 2 Objective 5 & Section 6 Required User Experience:
- At least two profile-specific reasons supporting the recommendation
- Product-by-product inclusion rationale (which investor constraints they serve)
- Main downside or trade-off of the recommended portfolio
- Explicit explanation of why plausible alternatives were limited or excluded
- Complete warning aggregation (emergency buffer, lock-in, liquidity, horizon, cost, feasibility)
"""
from typing import Dict, Any, List
import pandas as pd


def generate_explanation(
    profile: Dict[str, Any],
    opt_result: Dict[str, Any],
    proj_result: Dict[str, Any],
    products_df: pd.DataFrame,
    validation_warnings: List[str],
) -> Dict[str, Any]:
    """Generate comprehensive, deterministic, human-readable explanations for an investor recommendation."""
    horizon = int(profile["horizon_months"])
    risk_tol = float(profile["risk_tolerance_score"])
    liq_need = float(profile["liquidity_need_inr"])
    emerg = float(profile["emergency_fund_months"])
    pref = str(profile.get("preference", "none")).lower()
    existing = float(profile["existing_investment_inr"])
    sip = float(profile["monthly_sip_inr"])
    goal = float(profile["goal_amount_inr"])

    weights = opt_result["weights"]
    pids = opt_result["product_ids"]
    rb = opt_result["risk_bucket_breakdown"]
    w_risk = opt_result["weighted_risk_score"]
    w_ret = opt_result["weighted_assumed_return"]
    w_exp = opt_result["weighted_expense_ratio"]
    ann_cost = opt_result["estimated_annual_cost_inr"]

    prod_lookup = products_df.set_index("product_id").to_dict(orient="index")

    # 1. Profile-specific reasons supporting the recommendation (minimum 3 provided)
    profile_reasons: List[str] = []

    # Reason A: Horizon & Risk Bucket alignment
    profile_reasons.append(
        f"Horizon & Suitability Guardrail Fit: For your {horizon}-month '{profile['goal_label']}' goal "
        f"({rb['horizon_bucket']}) and Risk Tolerance Score of {risk_tol:.1f}/5.0 (implied Mental Accounting "
        f"risk aversion gamma={opt_result['implied_gamma']:.2f}), the portfolio allocates {rb['risk_1_2']*100:.1f}% "
        f"to low-risk defensive assets (guardrail min {rb['min_r12_limit']*100:.0f}%) and {rb['risk_4_5']*100:.1f}% "
        f"to high-growth equity (guardrail max {rb['max_r45_limit']*100:.0f}%), achieving a compliant weighted risk "
        f"score of {w_risk:.2f} (limit <= {opt_result['max_allowed_risk_score']:.2f})."
    )

    # Reason B: Liquidity, Lock-in, and Preference alignment
    pref_text = f"and honors your '{pref}' preference " if pref != "none" else ""
    profile_reasons.append(
        f"Liquidity, Lock-In & Preference Alignment: {opt_result['liquid_weight']*100:.1f}% of the portfolio is held "
        f"in zero-lock-in funds settling within 1-3 days (providing INR {opt_result['liquid_existing_inr']:,.0f} "
        f"immediate T=0 liquidity against your INR {liq_need:,.0f} requirement) {pref_text}while strictly excluding "
        f"any product with a lock-in exceeding {horizon} months."
    )

    # Reason C: Cost Efficiency & Downside Risk Control (Sortino / Expense optimization)
    profile_reasons.append(
        f"Cost-Aware & Downside-Protected Efficiency: By combining {opt_result['active_product_count']} distinct "
        f"product categories (all capped <= 40%), the allocation achieves a Sortino ratio of {opt_result['sortino_ratio']:.2f} "
        f"(downside deviation {opt_result['portfolio_downside_volatility']*100:.2f}%) at a low weighted annual expense ratio "
        f"of {w_exp*100:.2f}% (estimated INR {ann_cost:,.0f}/year on {opt_result['cost_basis_label']})."
    )

    # 2. Product-by-product inclusion rationale
    included_products_rationale: List[Dict[str, Any]] = []
    for i, pid in enumerate(pids):
        w = float(weights[i])
        if w <= 0.0:
            continue
        pinfo = prod_lookup[pid]
        role_parts = []
        if pinfo["risk_score"] <= 2 and pinfo["lock_in_months"] == 0:
            role_parts.append(
                f"Satisfies the {rb['min_r12_limit']*100:.0f}% Risk 1-2 floor and provides {pinfo['liquidity_days']}-day "
                f"zero-lock-in liquidity"
            )
        elif pinfo["risk_score"] <= 2 and pinfo["lock_in_months"] > 0:
            role_parts.append(
                f"Anchors the defensive debt bucket with {pinfo['assumed_annual_return']*100:.1f}% fixed return and "
                f"0.00% expense ratio ({pinfo['lock_in_months']}m lock-in <= {horizon}m horizon)"
            )
        elif pinfo["risk_score"] == 3:
            role_parts.append(
                "Bridges defensive debt and growth equity with balanced risk (Score 3) and moderate volatility (9.0%)"
            )
        elif pid == "PROD06":
            role_parts.append(
                "Captures broad-market equity compounding (11.0% assumed return) at an institutional low cost of 0.20% p.a."
            )
        elif pid == "PROD07":
            role_parts.append(
                f"Delivers 12.0% equity growth and tax-saving eligibility (36m lock-in fits within {horizon}m horizon)"
            )
        elif pid == "PROD08":
            role_parts.append(
                "Provides high-conviction thematic equity upside (13.5% assumed return) within the Risk 4-5 cap"
            )

        if pref == "low_cost" and pinfo["annual_expense_ratio"] <= 0.004:
            role_parts.append("directly serves your 'low_cost' preference")
        if pref == "tax_saving" and pid == "PROD07":
            role_parts.append("directly fulfills your 'tax_saving' preference")
        if emerg < 3.0 and pid in ("PROD01", "PROD03"):
            role_parts.append("doubles as an accessible emergency-buffer reserve")

        included_products_rationale.append({
            "product_id": pid,
            "product_name": pinfo["product_name"],
            "asset_class": pinfo["asset_class"],
            "weight_pct": round(w * 100, 2),
            "allocated_existing_inr": round(w * existing, 2),
            "allocated_sip_inr": round(w * sip, 2),
            "rationale": "; ".join(role_parts) + ".",
            "trade_off": (
                f"Assumed return {pinfo['assumed_annual_return']*100:.1f}%, Vol {pinfo['volatility_proxy']*100:.1f}%, "
                f"TER {pinfo['annual_expense_ratio']*100:.2f}%, Lock-in {pinfo['lock_in_months']}m."
            ),
        })

    # 3. Main portfolio downside or trade-off (Objective 5)
    if rb["max_r45_limit"] <= 0.10 or pref in ("avoid_equity", "capital_protection") or risk_tol <= 2:
        main_downside = (
            f"Opportunity Cost of Capital Preservation: Enforcing a {rb['risk_1_2']*100:.1f}% allocation in Risk 1-2 "
            f"cash/debt products caps the portfolio's gross expected return at {w_ret*100:.2f}% p.a. "
            f"(net {opt_result['weighted_net_return']*100:.2f}% after {w_exp*100:.2f}% fees). While this protects against "
            f"drawdowns over your {horizon}-month horizon, it limits upside compounding if equity markets rally."
        )
    elif w_risk >= 3.0:
        main_downside = (
            f"Interim Mark-to-Market Drawdown Risk: Holding {rb['risk_4_5']*100:.1f}% in Risk 4-5 equities and "
            f"{rb['risk_3']*100:.1f}% in hybrid funds exposes the portfolio to an annualized volatility of "
            f"{opt_result['portfolio_volatility']*100:.2f}% (downside volatility {opt_result['portfolio_downside_volatility']*100:.2f}%). "
            f"A 15% equity market correction would temporarily reduce initial portfolio value by "
            f"~{(opt_result['asset_class_breakdown']['equity']*15 + opt_result['asset_class_breakdown']['hybrid']*7.5):.1f}%."
        )
    else:
        main_downside = (
            f"Balanced Growth vs. Fee/Lock-In Trade-Off: The portfolio balances {w_ret*100:.2f}% expected return "
            f"against {opt_result['portfolio_volatility']*100:.2f}% volatility and a {w_exp*100:.2f}% weighted expense ratio. "
            f"Lock-in allocations (if any) cannot be liquidated early without penalty, requiring reliance on the "
            f"{opt_result['liquid_weight']*100:.1f}% liquid sleeve for interim cash needs."
        )

    # 4. Why plausible alternatives were limited or excluded (Objective 5)
    excluded_alternatives = [
        {"product_id": pid, "product_name": prod_lookup[pid]["product_name"], "reason": reason}
        for pid, reason in opt_result["exclusion_reasons"].items()
    ]

    # 5. Consolidate all warnings (Section 6)
    active_warnings = list(validation_warnings)
    if not proj_result["is_feasible"]:
        active_warnings.append(
            f"GOAL INFEASIBILITY WARNING: Base projected future value (INR {proj_result['fv_base_inr']:,.0f}) falls short "
            f"of target (INR {goal:,.0f}) by INR {abs(proj_result['surplus_shortfall_base_inr']):,.0f}. "
            f"Requires +INR {proj_result['additional_monthly_sip_needed_inr']:,.0f}/mo SIP or +{proj_result['additional_months_needed']}m horizon."
        )
    elif proj_result["surplus_shortfall_low_inr"] < 0:
        active_warnings.append(
            f"CONSERVATIVE SCENARIO SHORTFALL: Goal is met in the Base Case (+INR {proj_result['surplus_shortfall_base_inr']:,.0f}), "
            f"but experiences a shortfall of INR {abs(proj_result['surplus_shortfall_low_inr']):,.0f} under the -3% Conservative return stress."
        )

    if w_exp > 0.0075:
        active_warnings.append(
            f"COST DRAG NOTICE: Weighted annual expense ratio is {w_exp*100:.2f}% (INR {ann_cost:,.0f}/yr), "
            f"driven by active hybrid/equity funds."
        )

    if opt_result["used_fallback"]:
        active_warnings.append(f"FALLBACK ACTIVATED: {opt_result['fallback_reason']}")

    return {
        "profile_reasons": profile_reasons,
        "included_products_rationale": included_products_rationale,
        "main_downside_tradeoff": main_downside,
        "excluded_alternatives": excluded_alternatives,
        "active_warnings": active_warnings,
        "educational_disclaimer": (
            "EDUCATIONAL ESTIMATE DISCLAIMER: All investment products, returns, and projections in Smart Wealth "
            "are synthetic and for educational planning only. Assumed returns and simulated probabilities are not "
            "guaranteed outcomes or claims of real-world market performance."
        ),
    }
