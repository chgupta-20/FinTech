"""
High-Level Pipeline & Standardized Input/Output Contract for Smart Wealth.
Ensures Section 7 Generalisation Requirement:
Accepts any valid investor profile dictionary or CSV DataFrame and produces
both detailed recommendation dicts and the two required submission CSV files.
"""
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from .validator import (
    validate_product_catalogue,
    validate_investor_profile,
    validate_allocation_weights,
)
from .optimizer import optimize_portfolio
from .projections import evaluate_goal_projections, run_what_if_scenarios
from .explainability import generate_explanation


def recommend_for_investor(
    raw_profile: Dict[str, Any],
    products_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    include_what_if: bool = True,
) -> Dict[str, Any]:
    """
    Reusable single-investor recommendation function (Section 7 Generalisation Contract).
    Handles malformed inputs gracefully by returning a structured fallback response.
    """
    products_clean = validate_product_catalogue(products_df)
    is_valid, cleaned_profile, errors, warnings = validate_investor_profile(raw_profile)

    if not is_valid:
        # Return structured fallback for malformed profile (Section 3.1 Minimum Data Validation)
        fallback_weights = np.array([1.0] + [0.0] * (len(products_clean) - 1))
        return {
            "status": "INVALID_INPUT_FALLBACK",
            "is_valid": False,
            "investor_id": str(raw_profile.get("investor_id", "UNKNOWN")),
            "validation_errors": errors,
            "validation_warnings": warnings,
            "fallback_portfolio": {
                "product_id": products_clean.iloc[0]["product_id"],
                "product_name": products_clean.iloc[0]["product_name"],
                "weight": 1.0,
                "message": "Input profile failed minimum data validation. Capital parked 100% in PROD01 Liquid Cash Reserve Fund as safe fallback.",
            },
            "weights": fallback_weights,
            "product_ids": list(products_clean["product_id"]),
        }

    opt_result = optimize_portfolio(cleaned_profile, products_clean, returns_df)

    # Verify allocation weights per Section 3.1
    cat_ids = list(products_clean["product_id"])
    w_ok, w_errs = validate_allocation_weights(opt_result["weights"], opt_result["product_ids"], cat_ids)
    if not w_ok:
        warnings.extend(w_errs)

    proj_result = evaluate_goal_projections(cleaned_profile, opt_result)
    explanation = generate_explanation(cleaned_profile, opt_result, proj_result, products_clean, warnings)

    what_if_results = []
    if include_what_if:
        what_if_results = run_what_if_scenarios(
            cleaned_profile, products_clean, returns_df, optimize_portfolio
        )

    return {
        "status": "SUCCESS",
        "is_valid": True,
        "investor_id": cleaned_profile["investor_id"],
        "profile": cleaned_profile,
        "optimization": opt_result,
        "projections": proj_result,
        "explanation": explanation,
        "what_if_scenarios": what_if_results,
        "validation_errors": [],
        "validation_warnings": warnings,
    }


def generate_batch_recommendations(
    profiles_df: pd.DataFrame,
    products_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the Smart Wealth engine across a DataFrame of investor personas and generate:
    1. allocations_df: One row per client/product allocation (Deliverable 2a)
    2. summary_df: One row per client with projections, risk fit, costs, liquidity, warnings (Deliverable 2b)
    3. full_results: List of complete recommendation dictionaries
    """
    products_clean = validate_product_catalogue(products_df)
    prod_lookup = products_clean.set_index("product_id").to_dict(orient="index")

    allocation_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    for _, row in profiles_df.iterrows():
        raw_prof = row.to_dict()
        res = recommend_for_investor(raw_prof, products_clean, returns_df, include_what_if=True)
        full_results.append(res)

        if not res["is_valid"]:
            inv_id = res["investor_id"]
            for i, pid in enumerate(res["product_ids"]):
                pinfo = prod_lookup[pid]
                w = float(res["weights"][i])
                allocation_rows.append({
                    "investor_id": inv_id,
                    "goal_label": str(raw_prof.get("goal_label", "INVALID")),
                    "product_id": pid,
                    "product_name": pinfo["product_name"],
                    "asset_class": pinfo["asset_class"],
                    "risk_score": pinfo["risk_score"],
                    "weight_pct": round(w * 100, 2),
                    "weight_decimal": round(w, 4),
                    "allocated_existing_inr": 0.0,
                    "allocated_monthly_sip_inr": 0.0,
                    "assumed_annual_return": pinfo["assumed_annual_return"],
                    "annual_expense_ratio": pinfo["annual_expense_ratio"],
                    "liquidity_days": pinfo["liquidity_days"],
                    "lock_in_months": pinfo["lock_in_months"],
                    "selection_status": "FALLBACK_SAFE" if w > 0 else "EXCLUDED_INVALID_INPUT",
                })
            summary_rows.append({
                "investor_id": inv_id,
                "goal_label": str(raw_prof.get("goal_label", "INVALID")),
                "status": "INVALID_INPUT_FALLBACK",
                "warnings": " | ".join(res["validation_errors"]),
            })
            continue

        prof = res["profile"]
        opt = res["optimization"]
        proj = res["projections"]
        exp = res["explanation"]
        rb = opt["risk_bucket_breakdown"]
        ab = opt["asset_class_breakdown"]

        for i, pid in enumerate(opt["product_ids"]):
            pinfo = prod_lookup[pid]
            w = float(opt["weights"][i])
            allocation_rows.append({
                "investor_id": prof["investor_id"],
                "goal_label": prof["goal_label"],
                "product_id": pid,
                "product_name": pinfo["product_name"],
                "asset_class": pinfo["asset_class"],
                "risk_score": int(pinfo["risk_score"]),
                "weight_pct": round(w * 100, 2),
                "weight_decimal": round(w, 4),
                "allocated_existing_inr": float(opt["allocated_existing_inr"][i]),
                "allocated_monthly_sip_inr": float(opt["allocated_monthly_sip_inr"][i]),
                "assumed_annual_return": float(pinfo["assumed_annual_return"]),
                "annual_expense_ratio": float(pinfo["annual_expense_ratio"]),
                "liquidity_days": int(pinfo["liquidity_days"]),
                "lock_in_months": int(pinfo["lock_in_months"]),
                "selection_status": "RECOMMENDED" if w > 0 else "EXCLUDED",
            })

        summary_rows.append({
            "investor_id": prof["investor_id"],
            "goal_label": prof["goal_label"],
            "status": res["status"],
            "goal_amount_inr": prof["goal_amount_inr"],
            "horizon_months": prof["horizon_months"],
            "existing_investment_inr": prof["existing_investment_inr"],
            "monthly_sip_inr": prof["monthly_sip_inr"],
            "investor_risk_tolerance": prof["risk_tolerance_score"],
            "preference": prof["preference"],
            "weighted_portfolio_risk_score": opt["weighted_risk_score"],
            "max_allowed_risk_score": opt["max_allowed_risk_score"],
            "risk_fit_status": "COMPLIANT" if opt["weighted_risk_score"] <= opt["max_allowed_risk_score"] + 1e-4 else "VIOLATION",
            "risk_1_2_pct": round(rb["risk_1_2"] * 100, 2),
            "risk_3_pct": round(rb["risk_3"] * 100, 2),
            "risk_4_5_pct": round(rb["risk_4_5"] * 100, 2),
            "cash_pct": round(ab["cash"] * 100, 2),
            "debt_pct": round(ab["debt"] * 100, 2),
            "hybrid_pct": round(ab["hybrid"] * 100, 2),
            "equity_pct": round(ab["equity"] * 100, 2),
            "weighted_assumed_return_pct": round(opt["weighted_assumed_return"] * 100, 3),
            "weighted_net_return_pct": round(opt["weighted_net_return"] * 100, 3),
            "portfolio_volatility_pct": round(opt["portfolio_volatility"] * 100, 3),
            "portfolio_downside_vol_pct": round(opt["portfolio_downside_volatility"] * 100, 3),
            "sharpe_ratio": opt["sharpe_ratio"],
            "sortino_ratio": opt["sortino_ratio"],
            "weighted_annual_expense_ratio_pct": round(opt["weighted_expense_ratio"] * 100, 4),
            "estimated_annual_product_cost_inr": opt["estimated_annual_cost_inr"],
            "fv_conservative_low_inr": proj["fv_low_inr"],
            "fv_base_inr": proj["fv_base_inr"],
            "fv_optimistic_high_inr": proj["fv_high_inr"],
            "fv_net_of_fees_inr": proj["fv_net_of_fees_inr"],
            "estimated_surplus_shortfall_base_inr": proj["surplus_shortfall_base_inr"],
            "estimated_surplus_shortfall_low_inr": proj["surplus_shortfall_low_inr"],
            "estimated_surplus_shortfall_high_inr": proj["surplus_shortfall_high_inr"],
            "goal_feasibility_status": proj["feasibility_status"],
            "gbwm_goal_success_prob_pct": round(proj["prob_reach_goal"] * 100, 1),
            "required_monthly_sip_inr": proj["required_monthly_sip_inr"],
            "additional_sip_needed_inr": proj["additional_monthly_sip_needed_inr"],
            "required_horizon_months": proj["required_horizon_months"],
            "liquidity_need_inr": prof["liquidity_need_inr"],
            "liquid_allocation_pct": round(opt["liquid_weight"] * 100, 2),
            "liquid_existing_inr": opt["liquid_existing_inr"],
            "liquidity_status": opt["liquidity_status"],
            "emergency_fund_months": prof["emergency_fund_months"],
            "needs_emergency_buffer": opt["emergency_buffer_plan"]["needs_emergency_buffer"],
            "recommended_emergency_buffer_inr": opt["emergency_buffer_plan"]["recommended_buffer_inr"],
            "key_reasons": " || ".join(exp["profile_reasons"][:2]),
            "main_tradeoff": exp["main_downside_tradeoff"],
            "warnings": " || ".join(exp["active_warnings"]) if exp["active_warnings"] else "NONE",
        })

    allocations_df = pd.DataFrame(allocation_rows)
    summary_df = pd.DataFrame(summary_rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        allocations_df.to_csv(os.path.join(output_dir, "client_product_allocations.csv"), index=False)
        summary_df.to_csv(os.path.join(output_dir, "client_summary_projections.csv"), index=False)

    return allocations_df, summary_df, full_results
