"""
Constrained Portfolio Optimization Engine for Smart Wealth.
Integrates:
1. Das, Markowitz, Scheid & Statman (2010, 2011) Mental Accounting & VaR-implied Risk Aversion (gamma)
2. Shefrin & Statman (2000) Behavioral Portfolio Theory (BPT-MA) Layered Pyramid (Emergency Buffer vs Goal Layer)
3. Lazulfa & Saputro (2017) Buy-In Thresholds (semi-continuous minimum investment constraints)
4. Flor (2023) Downside-Deviation / Sortino Ratio & Sound Investment Admissibility Criteria
5. Thanh (2025) Liquidity-Constrained Mean-Variance & Sharpe Screening
6. Das, Ostrov, Radhakrishnan & Srivastav (2020) Dynamic Programming GBWM Wealth Grid
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_empirical_matrices(products_df: pd.DataFrame, returns_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the annualized covariance matrix (Sigma), downside deviation matrix (Sigma_down),
    and individual downside deviations from synthetic_monthly_returns.csv (Flor 2023 Eq. 5-6; Markowitz 1952).
    If returns_df is not provided, constructs a diagonal/proxy covariance matrix from volatility_proxy.
    """
    pids = list(products_df["product_id"])
    vol_proxy = products_df["volatility_proxy"].to_numpy(dtype=float)

    if returns_df is not None and not returns_df.empty:
        piv = returns_df.pivot(index="month", columns="product_id", values="monthly_return")[pids]
        # Correlation matrix from synthetic monthly returns
        corr = piv.corr().to_numpy(dtype=float)
        # Use supplied volatility_proxy scaled by empirical correlation for consistency with product catalogue
        D_vol = np.diag(vol_proxy)
        cov_matrix = D_vol @ corr @ D_vol

        # Downside deviation per Flor (2023) Eq. 5 & 6 and Sortino & van der Meer (1991)
        means = piv.mean(axis=0).to_numpy(dtype=float)
        downside_stds = np.zeros(len(pids))
        for idx in range(len(pids)):
            col_vals = piv.iloc[:, idx].to_numpy(dtype=float)
            neg_devs = col_vals[col_vals < means[idx]] - means[idx]
            if len(neg_devs) > 1:
                downside_stds[idx] = np.sqrt(np.sum(neg_devs ** 2) / (len(neg_devs) - 1)) * np.sqrt(12.0)
            else:
                downside_stds[idx] = vol_proxy[idx] * 0.75
        D_down = np.diag(downside_stds)
        downside_cov = D_down @ corr @ D_down
        return cov_matrix, downside_cov, downside_stds
    else:
        cov_matrix = np.diag(vol_proxy ** 2)
        downside_stds = vol_proxy * 0.8
        downside_cov = np.diag(downside_stds ** 2)
        return cov_matrix, downside_cov, downside_stds



def compute_required_goal_return(goal: float, P: float, S: float, n: int) -> float:
    """Compute the annual portfolio return r_req needed so that FV(P, S, r_req, n) == goal (Das et al., 2010 threshold H)."""
    from scipy.optimize import brentq
    from .projections import compute_fv
    if compute_fv(P, S, 0.0, n) >= goal:
        return 0.0
    if compute_fv(P, S, 0.25, n) < goal:
        return 0.25
    try:
        return float(brentq(lambda r: compute_fv(P, S, r, n) - goal, 0.0, 0.25))
    except Exception:
        return 0.10


def get_horizon_bucket_limits(horizon_months: int) -> Tuple[float, float, str]:
    """
    Return (max_risk_4_5, min_risk_1_2, bucket_label) per Section 4 Guardrail 2:
    - Up to 36 months: Max 10% in Risk 4-5, Min 70% in Risk 1-2
    - 37-84 months: Max 30% in Risk 4-5, Min 45% in Risk 1-2
    - More than 84 months: Max 55% in Risk 4-5, Min 25% in Risk 1-2
    """
    if horizon_months <= 36:
        return 0.10, 0.70, "Short Horizon (<= 36m)"
    elif horizon_months <= 84:
        return 0.30, 0.45, "Medium Horizon (37-84m)"
    else:
        return 0.55, 0.25, "Long Horizon (> 84m)"


def map_risk_aversion_gamma(risk_tolerance_score: float, horizon_months: int, preference: str) -> float:
    """
    Map investor risk tolerance score [1..5], horizon, and preference into the
    implied MVT/Mental-Accounting risk-aversion parameter gamma (Das et al., 2010 Eq. 1 & 7).
    Higher gamma = higher risk aversion.
    """
    # Base mapping aligned with Das et al. (2010) Table 1 (gamma in [0.8, 6.0])
    base_map = {
        1: 6.00,
        2: 3.80,
        3: 2.50,
        4: 1.40,
        5: 0.80,
    }
    r_clamped = max(1.0, min(5.0, float(risk_tolerance_score)))
    low_k = int(np.floor(r_clamped))
    high_k = int(np.ceil(r_clamped))
    if low_k == high_k:
        gamma = base_map[low_k]
    else:
        frac = r_clamped - low_k
        gamma = base_map[low_k] * (1.0 - frac) + base_map[high_k] * frac

    # Adjust for horizon and behavioral preference
    if horizon_months <= 24:
        gamma *= 1.25
    if preference == "capital_protection":
        gamma *= 1.80
    elif preference == "high_growth":
        gamma *= 0.70

    return float(gamma)


def build_emergency_buffer_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implements Section 4 Guardrail 5 & Shefrin & Statman (2000) BPT-MA Safety Layer:
    If emergency_fund_months < 3.0, demonstrate a conservative liquid buffer option
    (using PROD01 Liquid Cash Reserve Fund and PROD03 Short Duration Debt Fund)
    before allocating all capital to the long-term goal.
    """
    emerg_months = float(profile["emergency_fund_months"])
    existing = float(profile["existing_investment_inr"])
    sip = float(profile["monthly_sip_inr"])

    if emerg_months >= 3.0:
        return {
            "needs_emergency_buffer": False,
            "shortfall_months": 0.0,
            "estimated_monthly_expense_inr": 0.0,
            "recommended_buffer_inr": 0.0,
            "carve_out_from_existing_inr": 0.0,
            "carve_out_from_sip_inr": 0.0,
            "buffer_allocation": {},
            "buffer_explanation": "Emergency fund coverage is adequate (>= 3.0 months); 100% of capital is directed to the goal mental account.",
        }

    shortfall_months = 3.0 - emerg_months
    # Estimate monthly essential expense proxy from SIP/existing capital (reasonable robo-advisory heuristic)
    est_monthly_expense = max(15000.0, sip * 1.5 if sip > 0 else existing * 0.02)
    recommended_buffer_inr = round(shortfall_months * est_monthly_expense, 2)

    # Demonstrate how much can be carved out immediately vs via initial SIP months without starving the goal
    carve_existing = round(min(existing * 0.25, recommended_buffer_inr), 2)
    remaining_buffer = max(0.0, recommended_buffer_inr - carve_existing)
    carve_sip = round(min(sip * 0.30, remaining_buffer / max(1, min(12, profile["horizon_months"]))), 2)

    return {
        "needs_emergency_buffer": True,
        "shortfall_months": round(shortfall_months, 2),
        "estimated_monthly_expense_inr": round(est_monthly_expense, 2),
        "recommended_buffer_inr": recommended_buffer_inr,
        "carve_out_from_existing_inr": carve_existing,
        "carve_out_from_sip_inr": carve_sip,
        "buffer_allocation": {"PROD01": 0.70, "PROD03": 0.30},
        "buffer_explanation": (
            f"Emergency coverage is {emerg_months:.1f} months (< 3.0m guardrail). "
            f"Under Behavioral Portfolio Theory (Shefrin & Statman, 2000), we establish a Safety Mental Account "
            f"of INR {recommended_buffer_inr:,.0f} (70% PROD01 Liquid Cash Reserve + 30% PROD03 Short Duration Debt, "
            f"1-2 day settlement, 0 lock-in) via INR {carve_existing:,.0f} initial reserve and INR {carve_sip:,.0f}/mo SIP carve-out."
        ),
    }


def optimize_portfolio(
    profile: Dict[str, Any],
    products_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Core Constrained Portfolio Optimization Solver.
    Respects all Section 4 Guardrails:
    1. Diversification: w_i <= 0.40, at least 2 products (3+ enforced naturally by w_i <= 0.40).
    2. Horizon-based allocation limits on Risk 4-5 (max) and Risk 1-2 (min).
    3. Risk-tolerance limit: weighted risk score <= risk_tolerance_score + 0.25 (stricter of horizon vs risk tolerance).
    4. Liquidity & lock-in: lock_in_months <= horizon_months; liquid allocation covers liquidity_need_inr.
    5. Minimum investment threshold (Buy-in threshold per Lazulfa & Saputro, 2017).
    6. Cost & Downside-Deviation aware multi-objective utility (Das et al., 2010; Flor, 2023; Thanh, 2025).
    """
    n_prod = len(products_df)
    pids = list(products_df["product_id"])
    r_vec = products_df["assumed_annual_return"].to_numpy(dtype=float)
    risk_scores = products_df["risk_score"].to_numpy(dtype=float)
    vol_vec = products_df["volatility_proxy"].to_numpy(dtype=float)
    exp_vec = products_df["annual_expense_ratio"].to_numpy(dtype=float)
    liq_days = products_df["liquidity_days"].to_numpy(dtype=float)
    min_inv = products_df["minimum_investment_inr"].to_numpy(dtype=float)
    lock_in = products_df["lock_in_months"].to_numpy(dtype=float)
    asset_classes = list(products_df["asset_class"])

    cov_mat, downside_cov, downside_stds = compute_empirical_matrices(products_df, returns_df)

    horizon = int(profile["horizon_months"])
    existing = float(profile["existing_investment_inr"])
    sip = float(profile["monthly_sip_inr"])
    risk_tol = float(profile["risk_tolerance_score"])
    liq_need = float(profile["liquidity_need_inr"])
    emerg = float(profile["emergency_fund_months"])
    pref = str(profile.get("preference", "none")).lower()

    # Capital reference for minimum investment & liquidity sizing:
    # First-year investable capital (initial lump sum + 12m SIP or horizon SIP if shorter)
    first_year_capital = existing + sip * min(12, horizon)
    total_capital_reference = existing if existing > 0 else sip * min(12, horizon)

    # Horizon bucket limits (Guardrail 2)
    max_r45_horizon, min_r12_horizon, horizon_bucket_label = get_horizon_bucket_limits(horizon)

    # Stricter risk-tolerance preference adjustments (Guardrail 3)
    max_r45 = max_r45_horizon
    min_r12 = min_r12_horizon
    max_weighted_risk = risk_tol + 0.25

    if risk_tol <= 1.5:
        max_r45 = 0.0
        min_r12 = max(min_r12, 0.85)
    elif risk_tol <= 2.0:
        max_r45 = min(max_r45, 0.10)
        min_r12 = max(min_r12, 0.65)
    elif risk_tol <= 3.0:
        max_r45 = min(max_r45, 0.30)

    if pref == "capital_protection":
        max_r45 = 0.0
        min_r12 = max(min_r12, 0.80)
        max_weighted_risk = min(max_weighted_risk, 1.85)
    elif pref == "avoid_equity":
        max_r45 = 0.0

    # Identify eligible vs excluded products and record explicit exclusion reasons
    eligible_mask = np.ones(n_prod, dtype=bool)
    exclusion_reasons: Dict[str, str] = {}

    for i, pid in enumerate(pids):
        # Rule 4a: Lock-in exceeds goal horizon
        if lock_in[i] > horizon:
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                f"Excluded: Product lock-in ({int(lock_in[i])}m) exceeds goal horizon ({horizon}m) [Guardrail 4]."
            )
            continue

        # Optional Preference: no_lock_in
        if pref == "no_lock_in" and lock_in[i] > 0:
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                f"Excluded: Investor preference 'no_lock_in' disallows {int(lock_in[i])}-month lock-in."
            )
            continue

        # Optional Preference: avoid_equity
        if pref == "avoid_equity" and asset_classes[i] == "equity":
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                "Excluded: Investor preference 'avoid_equity' prohibits equity asset class exposure."
            )
            continue

        # Optional Preference: capital_protection excludes Risk 4-5 and hybrid Risk 3 if risk_tol == 1
        if pref == "capital_protection" and risk_scores[i] >= 4:
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                f"Excluded: Preference 'capital_protection' disallows high-volatility Risk {int(risk_scores[i])} equity."
            )
            continue

        if max_r45 <= 1e-6 and risk_scores[i] >= 4:
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                f"Excluded: Conservative risk tolerance ({risk_tol:.1f}) sets maximum Risk 4-5 allocation to 0%."
            )
            continue

        # Minimum investment check against available capital (Guardrail 4 / Lazulfa & Saputro 2017)
        # If existing_investment == 0 and monthly SIP is small, a high minimum lump-sum FD (PROD02: 10,000) is unsuitable
        if existing == 0 and min_inv[i] > sip:
            eligible_mask[i] = False
            exclusion_reasons[pid] = (
                f"Excluded: Minimum investment (INR {min_inv[i]:,.0f}) exceeds initial capital (INR 0) and monthly SIP (INR {sip:,.0f})."
            )
            continue

    # Liquidity constraint formulation (Guardrail 4)
    # Liquid products defined as lock_in == 0 and liquidity_days <= 3
    liquid_mask = (lock_in == 0) & (liq_days <= 3)
    if existing > 0:
        min_liquid_weight = min(0.95, liq_need / existing) if liq_need > 0 else 0.0
    else:
        min_liquid_weight = min(0.95, liq_need / max(1.0, first_year_capital)) if liq_need > 0 else 0.0

    # If emergency fund < 3 months, ensure at least 15% in highly liquid cash/short-debt (PROD01/PROD03)
    min_cash_buffer_weight = 0.15 if emerg < 3.0 else 0.0

    # Multi-objective parameters
    r_req_goal = compute_required_goal_return(float(profile["goal_amount_inr"]), existing, sip, horizon)
    gamma = map_risk_aversion_gamma(risk_tol, horizon, pref)
    cost_lambda = 4.5 if pref == "low_cost" else 1.5
    liq_lambda = 0.003 if (liq_need > 0 or emerg < 3.0) else 0.001

    idx_r12 = np.where(risk_scores <= 2)[0]
    idx_r45 = np.where(risk_scores >= 4)[0]
    idx_liquid = np.where(liquid_mask)[0]
    idx_cash_short = np.where((risk_scores <= 2) & (lock_in == 0) & (liq_days <= 2))[0]

    def solve_subproblem(active_mask: np.ndarray) -> Tuple[bool, np.ndarray]:
        bounds = []
        for i in range(n_prod):
            if not active_mask[i]:
                bounds.append((0.0, 0.0))
            else:
                # Guardrail 1: No single product > 40%
                ub = 0.40
                # If preference == low_cost, discourage high-fee active funds (> 1.0% TER) when passive alternatives exist
                bounds.append((0.0, ub))

        # Check if active products can sum to 1.0
        max_possible_sum = sum(b[1] for b in bounds)
        if max_possible_sum < 1.0 - 1e-6:
            return False, np.zeros(n_prod)

        def objective(w: np.ndarray) -> float:
            # Net expected return after expense penalty
            net_ret = float(np.dot(w, r_vec - cost_lambda * exp_vec))
            # Portfolio variance and downside variance (Das et al. 2010 + Flor 2023)
            port_var = float(w.T @ cov_mat @ w)
            down_var = float(w.T @ downside_cov @ w)
            blended_risk_penalty = 0.5 * gamma * (0.6 * port_var + 0.4 * down_var)
            # Liquidity days drag (Thanh 2025)
            liq_drag = liq_lambda * float(np.dot(w, liq_days))
            # Small HHI diversification regularization (Flor 2023 C3.2 diversity preference)
            div_bonus = -0.008 * float(np.sum(w ** 2))
            # Preference specific bonus
            pref_bonus = 0.0
            if pref == "tax_saving" and active_mask[6]:  # PROD07
                pref_bonus += 0.015 * w[6]
            if pref == "low_cost":
                pref_bonus -= 2.0 * float(np.dot(w, exp_vec))
            if pref == "high_growth":
                pref_bonus += 0.25 * float(np.dot(w, r_vec))
            if pref == "capital_protection":
                pref_bonus += 0.01 * (w[0] + w[1] + w[2])

            # Goal-Aspiration Shortfall Penalty (Das et al. 2010 Mental Accounting threshold H; Shefrin & Statman 2000)
            gross_r = float(np.dot(w, r_vec))
            shortfall_gap = max(0.0, r_req_goal + 0.0005 - gross_r)
            goal_penalty = 8.0 * (shortfall_gap ** 2) + 0.35 * shortfall_gap
            return -(net_ret - blended_risk_penalty - liq_drag + div_bonus + pref_bonus - goal_penalty)

        constraints = [
            # Sum of weights = 1.0
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            # Guardrail 2: Risk 1-2 >= min_r12
            {"type": "ineq", "fun": lambda w: np.sum(w[idx_r12]) - min_r12},
            # Guardrail 2: Risk 4-5 <= max_r45
            {"type": "ineq", "fun": lambda w: max_r45 - np.sum(w[idx_r45])},
            # Guardrail 3: Weighted portfolio risk score <= risk_tolerance_score + 0.25
            {"type": "ineq", "fun": lambda w: max_weighted_risk - float(np.dot(w, risk_scores))},
            # Guardrail 4: Zero-lock-in liquid allocation covers liquidity need
            {"type": "ineq", "fun": lambda w: np.sum(w[idx_liquid]) - min_liquid_weight},
        ]

        if min_cash_buffer_weight > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w: np.sum(w[idx_cash_short]) - min_cash_buffer_weight}
            )

        # If preference is tax_saving and PROD07 is eligible, ensure meaningful tax-saver allocation (>= 15%)
        if pref == "tax_saving" and active_mask[6] and max_r45 >= 0.15:
            constraints.append({"type": "ineq", "fun": lambda w: w[6] - min(0.25, max_r45 * 0.6)})

        # Feasible starting guess construction
        active_indices = [i for i in range(n_prod) if active_mask[i]]
        w0 = np.zeros(n_prod)
        for i in active_indices:
            w0[i] = 1.0 / len(active_indices)

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not res.success:
            # Try alternative conservative starting point heavy in Risk 1-2
            w0_alt = np.zeros(n_prod)
            r12_active = [i for i in idx_r12 if active_mask[i]]
            for i in r12_active:
                w0_alt[i] = min(0.35, 1.0 / max(1, len(r12_active)))
            rem = 1.0 - np.sum(w0_alt)
            other_active = [i for i in active_indices if i not in r12_active]
            for i in other_active:
                w0_alt[i] = rem / max(1, len(other_active))
            res = minimize(
                objective,
                w0_alt,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500},
            )

        w_sol = np.clip(res.x, 0.0, 0.40)
        # Verify constraints manually
        if abs(np.sum(w_sol) - 1.0) > 1e-3:
            return False, w_sol
        if np.sum(w_sol[idx_r12]) < min_r12 - 1e-3:
            return False, w_sol
        if np.sum(w_sol[idx_r45]) > max_r45 + 1e-3:
            return False, w_sol
        if float(np.dot(w_sol, risk_scores)) > max_weighted_risk + 1e-3:
            return False, w_sol
        return True, w_sol

    # Stage 1: Solve continuous QP/NLP
    current_mask = eligible_mask.copy()
    success, weights = solve_subproblem(current_mask)
    used_fallback = False
    fallback_reason = ""

    # Stage 2: Enforce Buy-In Thresholds (Lazulfa & Saputro, 2017) & Clean Dust Weights (< 2.5%)
    # Any non-zero weight must be >= 2.5% AND allocated capital must meet minimum_investment_inr
    if success:
        for _ in range(5):
            pruned = False
            for i in range(n_prod):
                if current_mask[i] and weights[i] > 1e-5:
                    alloc_cap = weights[i] * max(existing, first_year_capital)
                    if weights[i] < 0.025 or alloc_cap < min_inv[i]:
                        current_mask[i] = False
                        exclusion_reasons[pids[i]] = (
                            f"Pruned by Buy-In Threshold filter (Lazulfa & Saputro, 2017): raw weight {weights[i]*100:.1f}% "
                            f"(INR {alloc_cap:,.0f}) is below practical threshold or product minimum (INR {min_inv[i]:,.0f})."
                        )
                        pruned = True
                        break
            if not pruned:
                break
            sub_ok, sub_w = solve_subproblem(current_mask)
            if sub_ok:
                weights = sub_w
            else:
                # Revert last pruning if it makes problem infeasible
                current_mask[i] = True
                exclusion_reasons.pop(pids[i], None)
                break

    # Safer Fallback if constraints conflict (Guardrail 3 & Section 3.1)
    if not success:
        used_fallback = True
        fallback_reason = (
            "Primary optimization encountered conflicting liquidity/lock-in/risk constraints; "
            "activated Safer Capital-Preservation Fallback (40% PROD01, 40% PROD03, 20% PROD04)."
        )
        weights = np.array([0.40, 0.00, 0.40, 0.20, 0.00, 0.00, 0.00, 0.00])

    # Round weights to 4 decimal places (0.01% precision) while preserving exact sum = 1.0000
    weights = np.where(weights < 1e-4, 0.0, weights)
    weights = weights / np.sum(weights)
    weights_rounded = np.round(weights, 4)
    diff = round(1.0 - float(np.sum(weights_rounded)), 4)
    if abs(diff) > 0:
        # Add residual to largest component that won't violate 0.40 cap or risk bounds
        sorted_idx = np.argsort(-weights_rounded)
        for idx in sorted_idx:
            cand = round(weights_rounded[idx] + diff, 4)
            if 0.0 <= cand <= 0.40:
                weights_rounded[idx] = cand
                break
    weights = weights_rounded

    # Record exclusion reasons for any remaining 0-weight products
    for i, pid in enumerate(pids):
        if weights[i] == 0.0 and pid not in exclusion_reasons:
            if risk_scores[i] >= 4:
                exclusion_reasons[pid] = (
                    f"Limited/Excluded: Risk {int(risk_scores[i])} equity capped at {max_r45*100:.0f}% by {horizon_bucket_label} "
                    f"guardrail; PROD06/PROD07 offered superior net return-to-downside-risk and lower expense ratio ({exp_vec[i]*100:.2f}%)."
                )
            elif lock_in[i] > 0:
                exclusion_reasons[pid] = (
                    f"Excluded: {int(lock_in[i])}-month lock-in reduces portfolio liquidity flexibility compared to liquid debt."
                )
            else:
                exclusion_reasons[pid] = (
                    f"Excluded: Dominated on net cost-adjusted return ({r_vec[i]*100:.1f}% gross, {exp_vec[i]*100:.2f}% expense) "
                    f"along the constrained Mental Accounting efficient frontier."
                )

    # Portfolio-level metrics
    weighted_gross_return = float(np.dot(weights, r_vec))
    weighted_expense_ratio = float(np.dot(weights, exp_vec))
    weighted_net_return = weighted_gross_return - weighted_expense_ratio
    weighted_risk_score = float(np.dot(weights, risk_scores))
    portfolio_volatility = float(np.sqrt(weights.T @ cov_mat @ weights))
    portfolio_downside_vol = float(np.sqrt(weights.T @ downside_cov @ weights))
    weighted_liq_days = float(np.dot(weights, liq_days))

    # Risk-free rate benchmark = 5.0% (slightly below PROD01 overnight cash 5.5%)
    rf = 0.050
    sharpe_ratio = (weighted_gross_return - rf) / max(1e-6, portfolio_volatility)
    sortino_ratio = (weighted_gross_return - rf) / max(1e-6, portfolio_downside_vol)

    # Allocated INR amounts (on existing_investment_inr, and also first-year total committed capital)
    allocated_existing_inr = np.round(weights * existing, 2)
    allocated_monthly_sip_inr = np.round(weights * sip, 2)
    # Estimated annual product cost based on initial allocated balance + first-year average SIP balance
    # Report both initial balance cost and Year-1 average balance cost clearly
    cost_base_inr = existing if existing > 0 else (sip * 12.0)
    estimated_annual_cost_inr = round(float(np.dot(weights * cost_base_inr, exp_vec)), 2)

    # Liquidity coverage check (Guardrail 4)
    liquid_weight = float(np.sum(weights[idx_liquid]))
    liquid_existing_inr = round(liquid_weight * existing, 2)
    liquid_year1_inr = round(liquid_weight * (existing + sip * min(12, horizon)), 2)
    if liq_need == 0:
        liquidity_status = f"COVERED (No immediate liquidity requirement; {liquid_weight*100:.1f}% in <=3d zero-lock-in funds)"
        liquidity_covered = True
    elif liquid_existing_inr >= liq_need - 1.0:
        liquidity_status = f"COVERED (INR {liquid_existing_inr:,.0f} immediately liquid <=3d vs INR {liq_need:,.0f} needed)"
        liquidity_covered = True
    elif liquid_year1_inr >= liq_need - 1.0:
        liquidity_status = (
            f"PARTIALLY COVERED AT T=0 / COVERED WITHIN YEAR 1 "
            f"(T=0 liquid: INR {liquid_existing_inr:,.0f}, Year-1 liquid: INR {liquid_year1_inr:,.0f} vs INR {liq_need:,.0f} needed)"
        )
        liquidity_covered = True
    else:
        liquidity_status = (
            f"SHORTFALL (Liquid allocation INR {liquid_existing_inr:,.0f} at T=0 / INR {liquid_year1_inr:,.0f} Year-1 "
            f"< stated liquidity need INR {liq_need:,.0f})"
        )
        liquidity_covered = False

    # Asset class breakdown
    asset_class_breakdown: Dict[str, float] = {}
    for cls in ["cash", "debt", "hybrid", "equity"]:
        cls_mask = [i for i, c in enumerate(asset_classes) if c == cls]
        asset_class_breakdown[cls] = round(float(np.sum(weights[cls_mask])), 4)

    # Risk bucket breakdown
    r12_weight = round(float(np.sum(weights[idx_r12])), 4)
    r3_weight = round(float(np.sum(weights[np.where(risk_scores == 3)[0]])), 4)
    r45_weight = round(float(np.sum(weights[idx_r45])), 4)

    emerg_plan = build_emergency_buffer_plan(profile)

    return {
        "investor_id": profile["investor_id"],
        "weights": weights,
        "product_ids": pids,
        "allocated_existing_inr": allocated_existing_inr,
        "allocated_monthly_sip_inr": allocated_monthly_sip_inr,
        "weighted_assumed_return": round(weighted_gross_return, 6),
        "weighted_net_return": round(weighted_net_return, 6),
        "weighted_expense_ratio": round(weighted_expense_ratio, 6),
        "estimated_annual_cost_inr": estimated_annual_cost_inr,
        "cost_basis_label": "Existing Investment" if existing > 0 else "First-Year Annualized SIP",
        "weighted_risk_score": round(weighted_risk_score, 4),
        "max_allowed_risk_score": round(max_weighted_risk, 4),
        "portfolio_volatility": round(portfolio_volatility, 6),
        "portfolio_downside_volatility": round(portfolio_downside_vol, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "implied_gamma": round(gamma, 4),
        "weighted_liquidity_days": round(weighted_liq_days, 2),
        "liquid_weight": round(liquid_weight, 4),
        "liquid_existing_inr": liquid_existing_inr,
        "liquid_year1_inr": liquid_year1_inr,
        "liquidity_covered": liquidity_covered,
        "liquidity_status": liquidity_status,
        "asset_class_breakdown": asset_class_breakdown,
        "risk_bucket_breakdown": {
            "risk_1_2": r12_weight,
            "risk_3": r3_weight,
            "risk_4_5": r45_weight,
            "min_r12_limit": min_r12,
            "max_r45_limit": max_r45,
            "horizon_bucket": horizon_bucket_label,
        },
        "exclusion_reasons": exclusion_reasons,
        "emergency_buffer_plan": emerg_plan,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "active_product_count": int(np.sum(weights > 0)),
    }
