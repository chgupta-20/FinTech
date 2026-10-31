"""
Data Validation and Pre-processing Module for Smart Wealth.
Implements Section 3.1 Minimum Data Validation and Section 4 Suitability Guardrails.
"""
import math
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


REQUIRED_PROFILE_FIELDS = [
    "investor_id",
    "goal_label",
    "goal_amount_inr",
    "horizon_months",
    "existing_investment_inr",
    "monthly_sip_inr",
    "risk_tolerance_score",
    "emergency_fund_months",
    "liquidity_need_inr",
    "experience",
]

REQUIRED_PRODUCT_FIELDS = [
    "product_id",
    "product_name",
    "asset_class",
    "risk_score",
    "assumed_annual_return",
    "volatility_proxy",
    "annual_expense_ratio",
    "liquidity_days",
    "minimum_investment_inr",
    "lock_in_months",
]

VALID_PREFERENCES = {
    None, "", "none", "high_growth", "capital_protection",
    "low_cost", "avoid_equity", "tax_saving", "no_lock_in"
}

VALID_EXPERIENCES = {"beginner", "intermediate", "advanced"}


class ValidationError(Exception):
    """Custom exception raised when investor or product inputs fail validation."""
    pass


def validate_product_catalogue(products_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the supplied product catalogue has all required columns and finite values."""
    if not isinstance(products_df, pd.DataFrame) or products_df.empty:
        raise ValidationError("Product catalogue is empty or not a valid DataFrame.")

    missing_cols = [c for c in REQUIRED_PRODUCT_FIELDS if c not in products_df.columns]
    if missing_cols:
        raise ValidationError(f"Product catalogue missing required columns: {missing_cols}")

    if products_df["product_id"].isnull().any() or products_df["product_id"].duplicated().any():
        raise ValidationError("Product catalogue contains null or duplicate product_id values.")

    numeric_cols = [
        "risk_score", "assumed_annual_return", "volatility_proxy",
        "annual_expense_ratio", "liquidity_days", "minimum_investment_inr", "lock_in_months"
    ]
    for col in numeric_cols:
        vals = pd.to_numeric(products_df[col], errors="coerce")
        if vals.isnull().any() or not np.all(np.isfinite(vals)):
            raise ValidationError(f"Product catalogue column '{col}' contains non-numeric or non-finite values.")
        if (vals < 0).any():
            raise ValidationError(f"Product catalogue column '{col}' contains negative values.")

    return products_df.copy()


def validate_investor_profile(profile: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str], List[str]]:
    """
    Validate a single investor profile dictionary.
    Returns:
        is_valid (bool): True if valid (or recoverable with warnings), False if critically malformed.
        cleaned (dict): Cleaned and type-coerced profile dictionary.
        errors (List[str]): Critical validation errors triggering fallback/exception.
        warnings (List[str]): Suitability or contradiction warnings discovered during profiling.
    """
    errors: List[str] = []
    warnings: List[str] = []
    cleaned: Dict[str, Any] = {}

    if not isinstance(profile, dict):
        return False, {}, ["Profile input must be a dictionary or mapping."], []

    for field in REQUIRED_PROFILE_FIELDS:
        if field not in profile:
            errors.append(f"Missing required field: '{field}'.")

    if errors:
        return False, {}, errors, warnings

    cleaned["investor_id"] = str(profile.get("investor_id", "")).strip()
    cleaned["goal_label"] = str(profile.get("goal_label", "")).strip()
    if not cleaned["investor_id"]:
        errors.append("Field 'investor_id' cannot be empty.")

    numeric_Rules = {
        "goal_amount_inr": (1.0, 1e13, True),
        "horizon_months": (1.0, 1200.0, True),
        "existing_investment_inr": (0.0, 1e13, False),
        "monthly_sip_inr": (0.0, 1e11, False),
        "risk_tolerance_score": (1.0, 5.0, False),
        "emergency_fund_months": (0.0, 600.0, False),
        "liquidity_need_inr": (0.0, 1e13, False),
    }

    for col, (low, high, strictly_pos) in numeric_Rules.items():
        raw_val = profile.get(col)
        try:
            if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)) or str(raw_val).strip() == "":
                errors.append(f"Field '{col}' is null or missing.")
                continue
            val = float(raw_val)
            if not math.isfinite(val):
                errors.append(f"Field '{col}' must be finite (got {raw_val}).")
                continue
            if strictly_pos and val <= 0:
                errors.append(f"Field '{col}' must be strictly positive (> 0), got {val}.")
                continue
            if val < low or val > high:
                errors.append(f"Field '{col}'={val} is outside valid bounds [{low}, {high}].")
                continue
            cleaned[col] = val
        except (ValueError, TypeError):
            errors.append(f"Field '{col}' could not be parsed as a valid number: {raw_val}.")

    if not errors:
        cleaned["horizon_months"] = int(round(cleaned["horizon_months"]))
        if cleaned["existing_investment_inr"] == 0 and cleaned["monthly_sip_inr"] == 0:
            errors.append("Both existing_investment_inr and monthly_sip_inr are 0; no investable capital available.")

    # Experience check
    exp_raw = str(profile.get("experience", "intermediate")).strip().lower()
    if exp_raw not in VALID_EXPERIENCES:
        warnings.append(f"Unrecognized experience '{exp_raw}'; defaulting to 'beginner' for safety.")
        cleaned["experience"] = "beginner"
    else:
        cleaned["experience"] = exp_raw

    # Preference check
    pref_raw = profile.get("preference", None)
    if pref_raw is None or (isinstance(pref_raw, float) and math.isnan(pref_raw)):
        cleaned["preference"] = "none"
    else:
        pref_str = str(pref_raw).strip().lower()
        if pref_str in ("", "nan", "none", "null"):
            cleaned["preference"] = "none"
        elif pref_str not in VALID_PREFERENCES:
            warnings.append(f"Unknown preference '{pref_str}' ignored; using standard suitability rules.")
            cleaned["preference"] = "none"
        else:
            cleaned["preference"] = pref_str

    if errors:
        return False, cleaned, errors, warnings

    # Contradiction & Suitability Diagnostics (Objective 1 & Section 4)
    horizon = cleaned["horizon_months"]
    risk_tol = cleaned["risk_tolerance_score"]
    pref = cleaned["preference"]
    emerg = cleaned["emergency_fund_months"]
    liq_need = cleaned["liquidity_need_inr"]
    existing = cleaned["existing_investment_inr"]
    sip = cleaned["monthly_sip_inr"]

    if horizon <= 36 and (risk_tol >= 4 or pref == "high_growth"):
        warnings.append(
            f"HORIZON-RISK CONFLICT: Investor has high risk tolerance ({risk_tol:.1f}) or 'high_growth' preference, "
            f"but short goal horizon ({horizon}m <= 36m) enforces Guardrail 2 (Max 10% in Risk 4-5, Min 70% in Risk 1-2)."
        )

    if pref == "avoid_equity" and risk_tol >= 4:
        warnings.append(
            f"PREFERENCE-RISK CONTRADICTION: Risk tolerance is {risk_tol:.1f}/5.0, but preference 'avoid_equity' "
            f"excludes all Risk 4-5 equity products (PROD06, PROD07, PROD08), capping portfolio risk score at <= 2.35."
        )

    if pref == "capital_protection" and risk_tol >= 3:
        warnings.append(
            f"CAPITAL PROTECTION OVERRIDE: Preference 'capital_protection' enforces conservative preservation "
            f"despite moderate/high risk tolerance ({risk_tol:.1f})."
        )

    if pref == "tax_saving" and horizon < 36:
        warnings.append(
            f"TAX-SAVING LOCK-IN CONFLICT: Preference 'tax_saving' requests PROD07 (36m lock-in), "
            f"which is disallowed because goal horizon is {horizon}m (< 36m)."
        )

    if emerg < 3.0:
        shortfall_months = 3.0 - emerg
        warnings.append(
            f"EMERGENCY BUFFER WARNING: Emergency fund coverage is {emerg:.1f} months (< 3.0 months minimum). "
            f"Recommend building {shortfall_months:.1f} additional months of liquid reserve in PROD01/PROD03."
        )

    total_contrib = existing + sip * horizon
    if liq_need > existing and existing > 0:
        warnings.append(
            f"LIQUIDITY VS INITIAL CAPITAL: Stated liquidity need (INR {liq_need:,.0f}) exceeds initial lump sum "
            f"(INR {existing:,.0f}); ongoing SIP contributions into liquid funds are required to cover liquidity."
        )
    elif liq_need > total_contrib:
        warnings.append(
            f"LIQUIDITY EXCEEDS CAPITAL: Stated liquidity need (INR {liq_need:,.0f}) exceeds total contributions "
            f"(INR {total_contrib:,.0f})."
        )

    return True, cleaned, errors, warnings


def validate_allocation_weights(weights: np.ndarray, product_ids: List[str], catalogue_ids: List[str], tol: float = 0.001) -> Tuple[bool, List[str]]:
    """
    Validate portfolio weights per Section 3.1 Minimum Data Validation:
    - Non-negative weights
    - Sum to 100% within 0.1 percentage points (0.001 in decimal)
    - All referenced product_ids exist in supplied catalogue
    - Finite numeric values
    """
    errs = []
    if len(weights) != len(product_ids):
        errs.append("Length of weights array does not match product_ids.")
        return False, errs

    for pid in product_ids:
        if pid not in catalogue_ids:
            errs.append(f"Referenced product_id '{pid}' does not exist in catalogue.")

    if not np.all(np.isfinite(weights)):
        errs.append("Allocation weights contain non-finite (NaN or Inf) values.")
        return False, errs

    if np.any(weights < -1e-7):
        errs.append(f"Allocation weights contain negative values: min={np.min(weights):.6f}.")

    w_sum = float(np.sum(weights))
    if abs(w_sum - 1.0) > tol:
        errs.append(f"Allocation weights sum to {w_sum*100:.4f}%, outside 100% +/- {tol*100:.2f}% tolerance.")

    return len(errs) == 0, errs
