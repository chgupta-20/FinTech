"""
Goal Projection, Stochastic GBWM Probability, Recourse Analysis, and What-If Engine.
Implements:
1. Section 5 Deterministic Goal Projection Formula:
   FV(P, S, r, n) = P*(1 + r/12)^n + S*(((1 + r/12)^n - 1) / (r/12))
   with scenario range [r_p - 3%, r_p, r_p + 3%] and floor at 0.5%.
2. Das, Ostrov, Radhakrishnan & Srivastav (2020) Analytical & Monte Carlo Goal Success Probability Pr[W(T) >= G]
   plus Roy (1952) / Kataoka (1963) Safety-First Capital Preservation Probability Pr[W(T) >= Principal].
3. Actionable Shortfall Recourse Calculator (Required Monthly SIP, Required Horizon Extension, Required Lump Sum).
4. Objective 6 Multi-Scenario What-If Stress Engine (15% Market Shock, 20% SIP Reduction, Horizon Shortening, Risk Shift).
"""
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def compute_fv(P: float, S: float, r: float, n: int) -> float:
    """
    Section 5 Goal Projection Method:
    FV(P, S, r, n) = P * (1 + r/12)^n + S * (((1 + r/12)^n - 1) / (r/12))
    When abs(r) < 1e-9, FV = P + S * n.
    """
    P = float(P)
    S = float(S)
    r = float(r)
    n = int(n)
    if n <= 0:
        return round(P, 2)
    if abs(r) < 1e-9:
        return round(P + S * n, 2)
    r_m = r / 12.0
    compound = (1.0 + r_m) ** n
    fv = P * compound + S * ((compound - 1.0) / r_m)
    return round(float(fv), 2)


def solve_required_sip(goal: float, P: float, r: float, n: int) -> float:
    """Solve algebraically for the monthly SIP required to reach `goal` at annual rate `r` in `n` months."""
    if n <= 0:
        return 0.0
    if abs(r) < 1e-9:
        req = (goal - P) / n
        return round(max(0.0, req), 2)
    r_m = r / 12.0
    compound = (1.0 + r_m) ** n
    future_p = P * compound
    if future_p >= goal:
        return 0.0
    annuity_factor = (compound - 1.0) / r_m
    req_sip = (goal - future_p) / max(1e-9, annuity_factor)
    return round(max(0.0, float(req_sip)), 2)


def solve_required_horizon_months(goal: float, P: float, S: float, r: float, max_months: int = 600) -> int:
    """Find the minimum number of months required to reach `goal` given initial `P`, monthly `S`, and rate `r`."""
    for m in range(1, max_months + 1):
        if compute_fv(P, S, r, m) >= goal:
            return m
    return max_months


def simulate_gbwm_probabilities(
    P: float,
    S: float,
    mu: float,
    sigma: float,
    n_months: int,
    goal: float,
    n_sims: int = 2500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute Goal Achievement Probability Pr[W(T) >= G] and Capital Preservation Probability Pr[W(T) >= Principal]
    using Geometric Brownian Motion with periodic cash infusions (Das, Ostrov et al. 2020 Eq. 2-4; Flor 2023 Section 3.7).
    Educational estimate only; not a forecast guarantee.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    vol_step = sigma * math.sqrt(dt)

    # Vectorized simulation across n_sims paths over n_months
    shocks = rng.standard_normal((n_sims, n_months))
    growth_factors = np.exp(drift + vol_step * shocks)

    wealth = np.full(n_sims, float(P), dtype=float)
    for t in range(n_months):
        wealth = wealth * growth_factors[:, t] + float(S)

    total_principal = float(P + S * n_months)
    prob_goal = float(np.mean(wealth >= goal))
    prob_preserve = float(np.mean(wealth >= total_principal))
    p10 = float(np.percentile(wealth, 10))
    p50 = float(np.percentile(wealth, 50))
    p90 = float(np.percentile(wealth, 90))

    return {
        "prob_reach_goal": round(prob_goal, 4),
        "prob_preserve_principal": round(prob_preserve, 4),
        "mc_p10_inr": round(p10, 2),
        "mc_median_inr": round(p50, 2),
        "mc_p90_inr": round(p90, 2),
    }


def evaluate_goal_projections(
    profile: Dict[str, Any],
    opt_result: Dict[str, Any],
    return_floor: float = 0.005,
) -> Dict[str, Any]:
    """
    Evaluate goal projections under Section 5 formula:
    - Base scenario: r_p (weighted assumed annual return)
    - Low (Conservative) scenario: max(return_floor, r_p - 0.03)
    - High (Optimistic) scenario: r_p + 0.03
    - Net-of-fee base scenario: r_p - weighted_expense_ratio
    Also computes surplus/shortfall in INR and actionable recourse recommendations.
    """
    P = float(profile["existing_investment_inr"])
    S = float(profile["monthly_sip_inr"])
    n = int(profile["horizon_months"])
    goal = float(profile["goal_amount_inr"])

    r_p = float(opt_result["weighted_assumed_return"])
    e_p = float(opt_result["weighted_expense_ratio"])
    sigma_p = float(opt_result["portfolio_volatility"])

    # Section 5 scenario rates: r_p - 3 percentage points, r_p, r_p + 3 percentage points
    # Floor at 0.5% (0.005) explained clearly so short-term debt scenarios remain realistic positive nominal rates
    r_low = max(return_floor, round(r_p - 0.03, 6))
    r_base = round(r_p, 6)
    r_high = round(r_p + 0.03, 6)
    r_net = max(0.0, round(r_p - e_p, 6))

    fv_low = compute_fv(P, S, r_low, n)
    fv_base = compute_fv(P, S, r_base, n)
    fv_high = compute_fv(P, S, r_high, n)
    fv_net = compute_fv(P, S, r_net, n)
    total_principal = round(P + S * n, 2)

    surplus_shortfall_base = round(fv_base - goal, 2)
    surplus_shortfall_low = round(fv_low - goal, 2)
    surplus_shortfall_high = round(fv_high - goal, 2)
    surplus_shortfall_net = round(fv_net - goal, 2)

    if fv_base >= goal and fv_low >= goal:
        feasibility_status = "FEASIBLE (Robust across Conservative & Base Scenarios)"
        is_feasible = True
    elif fv_base >= goal:
        feasibility_status = "FEASIBLE (Meets Goal in Base Scenario; Shortfall under -3% Stress)"
        is_feasible = True
    elif fv_high >= goal:
        feasibility_status = "MARGINAL / AT-RISK (Shortfall in Base Scenario; Achievable only in +3% Optimistic Case)"
        is_feasible = False
    else:
        feasibility_status = "INFEASIBLE UNDER CURRENT INPUTS (Shortfall across all Return Scenarios)"
        is_feasible = False

    # Recourse calculations when base FV < goal
    req_sip_base = solve_required_sip(goal, P, r_base, n)
    additional_sip_needed = round(max(0.0, req_sip_base - S), 2)
    req_horizon_months = solve_required_horizon_months(goal, P, S, r_base, max_months=600)
    additional_months_needed = max(0, req_horizon_months - n)

    if is_feasible:
        recourse_recommendation = (
            f"Goal is projected to be met in the Base Case with an estimated surplus of INR {surplus_shortfall_base:,.0f} "
            f"(Projected FV: INR {fv_base:,.0f} vs Target: INR {goal:,.0f}). Maintain monthly SIP of INR {S:,.0f} "
            f"and rebalance annually."
        )
    else:
        shortfall_abs = abs(surplus_shortfall_base)
        recourse_recommendation = (
            f"Estimated shortfall of INR {shortfall_abs:,.0f} in Base Case (Projected FV: INR {fv_base:,.0f} vs Target: INR {goal:,.0f}). "
            f"Practical Recourse Options (without violating risk guardrails): "
            f"(1) Increase monthly SIP by INR {additional_sip_needed:,.0f} (to INR {req_sip_base:,.0f}/month), OR "
            f"(2) Extend goal horizon by {additional_months_needed} months (to {req_horizon_months} months total), OR "
            f"(3) Revise target goal amount to ~INR {fv_base:,.0f}."
        )

    gbwm_probs = simulate_gbwm_probabilities(P, S, r_base, sigma_p, n, goal)

    # Monthly deterministic trajectory for charting
    months_arr = list(range(0, n + 1))
    traj_low = [compute_fv(P, S, r_low, m) for m in months_arr]
    traj_base = [compute_fv(P, S, r_base, m) for m in months_arr]
    traj_high = [compute_fv(P, S, r_high, m) for m in months_arr]
    traj_principal = [round(P + S * m, 2) for m in months_arr]

    return {
        "r_low": r_low,
        "r_base": r_base,
        "r_high": r_high,
        "r_net": r_net,
        "return_floor_used": return_floor,
        "fv_low_inr": fv_low,
        "fv_base_inr": fv_base,
        "fv_high_inr": fv_high,
        "fv_net_of_fees_inr": fv_net,
        "total_principal_contributed_inr": total_principal,
        "goal_amount_inr": round(goal, 2),
        "surplus_shortfall_base_inr": surplus_shortfall_base,
        "surplus_shortfall_low_inr": surplus_shortfall_low,
        "surplus_shortfall_high_inr": surplus_shortfall_high,
        "surplus_shortfall_net_inr": surplus_shortfall_net,
        "is_feasible": is_feasible,
        "feasibility_status": feasibility_status,
        "required_monthly_sip_inr": req_sip_base,
        "additional_monthly_sip_needed_inr": additional_sip_needed,
        "required_horizon_months": req_horizon_months,
        "additional_months_needed": additional_months_needed,
        "recourse_recommendation": recourse_recommendation,
        "prob_reach_goal": gbwm_probs["prob_reach_goal"],
        "prob_preserve_principal": gbwm_probs["prob_preserve_principal"],
        "mc_p10_inr": gbwm_probs["mc_p10_inr"],
        "mc_median_inr": gbwm_probs["mc_median_inr"],
        "mc_p90_inr": gbwm_probs["mc_p90_inr"],
        "trajectory": {
            "months": months_arr,
            "principal": traj_principal,
            "fv_low": traj_low,
            "fv_base": traj_base,
            "fv_high": traj_high,
        },
        "assumptions_note": (
            "Educational estimate using Section 5 monthly compounding approximation FV(P,S,r,n). "
            "Assumes monthly SIPs are contributed at the end of each month, annual return r_p is compounded at r_p/12 monthly, "
            "and low/high scenarios use r_p +/- 3.0% with a 0.5% nominal floor. Projections are not guaranteed outcomes."
        ),
    }


def run_what_if_scenarios(
    profile: Dict[str, Any],
    products_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame],
    optimizer_fn,
) -> List[Dict[str, Any]]:
    """
    Implements Section 2 Objective 6 (What-if Analysis):
    Demonstrates how recommendations and projections respond to 4 standardized stress scenarios:
    1. 15% Initial Market Shock on Equity/Hybrid holdings (Drawdown stress per Thanh 2025)
    2. 20% Reduction in Monthly SIP (Cash-flow infusion drop per Das et al. 2020)
    3. Shorter Goal Horizon (-12 months or -25%, triggering horizon guardrail transitions)
    4. Conservative Shift in Risk Tolerance (-1 point, min 1.0)
    """
    base_opt = optimizer_fn(profile, products_df, returns_df)
    base_proj = evaluate_goal_projections(profile, base_opt)

    scenarios: List[Dict[str, Any]] = []

    # Scenario 0: Baseline
    scenarios.append({
        "scenario_id": "BASELINE",
        "scenario_name": "Baseline Plan",
        "description": "Unmodified investor profile and market assumptions.",
        "weighted_risk_score": base_opt["weighted_risk_score"],
        "weighted_return_pct": round(base_opt["weighted_assumed_return"] * 100, 2),
        "weighted_expense_pct": round(base_opt["weighted_expense_ratio"] * 100, 3),
        "fv_base_inr": base_proj["fv_base_inr"],
        "surplus_shortfall_inr": base_proj["surplus_shortfall_base_inr"],
        "prob_reach_goal_pct": round(base_proj["prob_reach_goal"] * 100, 1),
        "required_sip_inr": base_proj["required_monthly_sip_inr"],
        "feasibility_status": base_proj["feasibility_status"],
        "top_allocations": _format_top_alloc(base_opt["weights"], base_opt["product_ids"]),
    })

    # Scenario 1: 15% Market Shock on Equity (and 7.5% on Hybrid) applied to existing investment
    w_base = base_opt["weights"]
    eq_weight = base_opt["asset_class_breakdown"]["equity"]
    hyb_weight = base_opt["asset_class_breakdown"]["hybrid"]
    portfolio_shock_pct = eq_weight * 0.15 + hyb_weight * 0.075
    p_shocked = dict(profile)
    p_shocked["existing_investment_inr"] = round(float(profile["existing_investment_inr"]) * (1.0 - portfolio_shock_pct), 2)
    proj_shock = evaluate_goal_projections(p_shocked, base_opt)
    scenarios.append({
        "scenario_id": "MARKET_SHOCK_15",
        "scenario_name": "15% Equity Market Shock",
        "description": (
            f"Immediate -15% shock to Equity and -7.5% to Hybrid reduces initial capital by "
            f"{portfolio_shock_pct*100:.2f}% (to INR {p_shocked['existing_investment_inr']:,.0f})."
        ),
        "weighted_risk_score": base_opt["weighted_risk_score"],
        "weighted_return_pct": round(base_opt["weighted_assumed_return"] * 100, 2),
        "weighted_expense_pct": round(base_opt["weighted_expense_ratio"] * 100, 3),
        "fv_base_inr": proj_shock["fv_base_inr"],
        "surplus_shortfall_inr": proj_shock["surplus_shortfall_base_inr"],
        "prob_reach_goal_pct": round(proj_shock["prob_reach_goal"] * 100, 1),
        "required_sip_inr": proj_shock["required_monthly_sip_inr"],
        "feasibility_status": proj_shock["feasibility_status"],
        "top_allocations": _format_top_alloc(w_base, base_opt["product_ids"]),
    })

    # Scenario 2: 20% Reduction in Monthly SIP
    p_sip_cut = dict(profile)
    p_sip_cut["monthly_sip_inr"] = round(float(profile["monthly_sip_inr"]) * 0.80, 2)
    opt_sip_cut = optimizer_fn(p_sip_cut, products_df, returns_df)
    proj_sip_cut = evaluate_goal_projections(p_sip_cut, opt_sip_cut)
    scenarios.append({
        "scenario_id": "SIP_CUT_20",
        "scenario_name": "20% Reduction in Monthly SIP",
        "description": f"Monthly SIP reduced by 20% from INR {profile['monthly_sip_inr']:,.0f} to INR {p_sip_cut['monthly_sip_inr']:,.0f}.",
        "weighted_risk_score": opt_sip_cut["weighted_risk_score"],
        "weighted_return_pct": round(opt_sip_cut["weighted_assumed_return"] * 100, 2),
        "weighted_expense_pct": round(opt_sip_cut["weighted_expense_ratio"] * 100, 3),
        "fv_base_inr": proj_sip_cut["fv_base_inr"],
        "surplus_shortfall_inr": proj_sip_cut["surplus_shortfall_base_inr"],
        "prob_reach_goal_pct": round(proj_sip_cut["prob_reach_goal"] * 100, 1),
        "required_sip_inr": proj_sip_cut["required_monthly_sip_inr"],
        "feasibility_status": proj_sip_cut["feasibility_status"],
        "top_allocations": _format_top_alloc(opt_sip_cut["weights"], opt_sip_cut["product_ids"]),
    })

    # Scenario 3: Shorter Goal Horizon (-12 months, or -6 months if horizon <= 18)
    p_short = dict(profile)
    delta_h = 12 if int(profile["horizon_months"]) > 24 else max(3, int(profile["horizon_months"]) // 4)
    p_short["horizon_months"] = max(6, int(profile["horizon_months"]) - delta_h)
    opt_short = optimizer_fn(p_short, products_df, returns_df)
    proj_short = evaluate_goal_projections(p_short, opt_short)
    scenarios.append({
        "scenario_id": "SHORTER_HORIZON",
        "scenario_name": f"Shorter Goal Horizon (-{delta_h}m to {p_short['horizon_months']}m)",
        "description": (
            f"Goal horizon shortened from {profile['horizon_months']}m to {p_short['horizon_months']}m, "
            f"re-running lock-in and horizon bucket guardrails."
        ),
        "weighted_risk_score": opt_short["weighted_risk_score"],
        "weighted_return_pct": round(opt_short["weighted_assumed_return"] * 100, 2),
        "weighted_expense_pct": round(opt_short["weighted_expense_ratio"] * 100, 3),
        "fv_base_inr": proj_short["fv_base_inr"],
        "surplus_shortfall_inr": proj_short["surplus_shortfall_base_inr"],
        "prob_reach_goal_pct": round(proj_short["prob_reach_goal"] * 100, 1),
        "required_sip_inr": proj_short["required_monthly_sip_inr"],
        "feasibility_status": proj_short["feasibility_status"],
        "top_allocations": _format_top_alloc(opt_short["weights"], opt_short["product_ids"]),
    })

    # Scenario 4: Risk Tolerance Shift (-1 point if risk_tol > 1, else +1 point)
    p_risk = dict(profile)
    old_rt = float(profile["risk_tolerance_score"])
    new_rt = max(1.0, old_rt - 1.0) if old_rt > 1.0 else min(5.0, old_rt + 1.0)
    p_risk["risk_tolerance_score"] = new_rt
    opt_risk = optimizer_fn(p_risk, products_df, returns_df)
    proj_risk = evaluate_goal_projections(p_risk, opt_risk)
    scenarios.append({
        "scenario_id": "RISK_TOLERANCE_SHIFT",
        "scenario_name": f"Risk Tolerance Changed ({old_rt:.0f} -> {new_rt:.0f})",
        "description": f"Investor risk tolerance score changed from {old_rt:.1f} to {new_rt:.1f}, re-optimizing portfolio weights.",
        "weighted_risk_score": opt_risk["weighted_risk_score"],
        "weighted_return_pct": round(opt_risk["weighted_assumed_return"] * 100, 2),
        "weighted_expense_pct": round(opt_risk["weighted_expense_ratio"] * 100, 3),
        "fv_base_inr": proj_risk["fv_base_inr"],
        "surplus_shortfall_inr": proj_risk["surplus_shortfall_base_inr"],
        "prob_reach_goal_pct": round(proj_risk["prob_reach_goal"] * 100, 1),
        "required_sip_inr": proj_risk["required_monthly_sip_inr"],
        "feasibility_status": proj_risk["feasibility_status"],
        "top_allocations": _format_top_alloc(opt_risk["weights"], opt_risk["product_ids"]),
    })

    return scenarios


def _format_top_alloc(weights: np.ndarray, pids: List[str]) -> str:
    pairs = [(pids[i], weights[i]) for i in range(len(pids)) if weights[i] > 0.001]
    pairs.sort(key=lambda x: -x[1])
    return ", ".join(f"{pid}:{w*100:.1f}%" for pid, w in pairs)
