"""
Command-Line Interface & Standardized Evaluation Contract for Smart Wealth.
Usage:
    python cli.py --profiles data/investor_profiles.csv --products data/investment_products.csv --returns data/synthetic_monthly_returns.csv --out_dir outputs/
"""
import argparse
import os
import sys
import pandas as pd
from src.engine import generate_batch_recommendations


def main():
    parser = argparse.ArgumentParser(description="Smart Wealth Goal-Based Investment Strategy CLI")
    parser.add_argument("--profiles", type=str, default="data/investor_profiles.csv", help="Path to investor profiles CSV")
    parser.add_argument("--products", type=str, default="data/investment_products.csv", help="Path to investment products CSV")
    parser.add_argument("--returns", type=str, default="data/synthetic_monthly_returns.csv", help="Path to synthetic monthly returns CSV")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to write output CSVs")
    parser.add_argument("--investor_id", type=str, default=None, help="Optional single investor_id to inspect in terminal")
    args = parser.parse_args()

    if not os.path.exists(args.profiles):
        print(f"Error: Profiles file not found at {args.profiles}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.products):
        print(f"Error: Products file not found at {args.products}", file=sys.stderr)
        sys.exit(1)

    profiles_df = pd.read_csv(args.profiles)
    products_df = pd.read_csv(args.products)
    returns_df = pd.read_csv(args.returns) if os.path.exists(args.returns) else None

    if args.investor_id:
        profiles_df = profiles_df[profiles_df["investor_id"] == args.investor_id]
        if profiles_df.empty:
            print(f"Error: Investor ID '{args.investor_id}' not found in {args.profiles}", file=sys.stderr)
            sys.exit(1)

    alloc_df, sum_df, full_results = generate_batch_recommendations(
        profiles_df, products_df, returns_df, output_dir=args.out_dir
    )

    print(f"[Smart Wealth] Processed {len(full_results)} investor profile(s).")
    print(f"[Smart Wealth] Saved allocations -> {os.path.join(args.out_dir, 'client_product_allocations.csv')}")
    print(f"[Smart Wealth] Saved summary     -> {os.path.join(args.out_dir, 'client_summary_projections.csv')}\n")

    for res in full_results:
        if not res["is_valid"]:
            print(f"=== {res['investor_id']} [INVALID INPUT FALLBACK] ===")
            print("Errors:", "; ".join(res["validation_errors"]))
            continue
        p = res["profile"]
        o = res["optimization"]
        pr = res["projections"]
        ex = res["explanation"]
        print(f"=== {p['investor_id']}: {p['goal_label']} (Horizon: {p['horizon_months']}m | Goal: INR {p['goal_amount_inr']:,.0f}) ===")
        print(f"  Risk Score: {o['weighted_risk_score']:.2f} (Limit <= {o['max_allowed_risk_score']:.2f}) | "
              f"Assumed Return: {o['weighted_assumed_return']*100:.2f}% | Expense Ratio: {o['weighted_expense_ratio']*100:.2f}% "
              f"(INR {o['estimated_annual_cost_inr']:,.0f}/yr)")
        print(f"  Projected FV (Low / Base / High): INR {pr['fv_low_inr']:,.0f} / INR {pr['fv_base_inr']:,.0f} / INR {pr['fv_high_inr']:,.0f}")
        print(f"  Surplus/Shortfall (Base): INR {pr['surplus_shortfall_base_inr']:+,.0f} | Status: {pr['feasibility_status']}")
        if ex["active_warnings"]:
            print(f"  Warnings ({len(ex['active_warnings'])}): " + " | ".join(ex["active_warnings"]))
        print()


if __name__ == "__main__":
    main()