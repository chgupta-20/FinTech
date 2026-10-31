# Smart Wealth: Goal-Based Investment Strategy Engine
**Inter-IIT Tech Meet 15.0 Practice Problem Statement — Complete Submission Package**

`Smart Wealth` is an explainable, cost-aware, and guardrail-compliant Goal-Based Wealth Management (GBWM) prototype built for the fictional Indian digital wealth platform **Nivesh Now**. The system converts an investor's financial goal profile and a catalogue of fictional investment products into a mathematically rigorous, suitability-compliant investment plan with deterministic projections, Monte Carlo goal-attainment probabilities, algebraic shortfall recourse plans, and multi-scenario What-If stress analysis.

---

## 1. Quick Start & Launch Commands

### Prerequisites
Ensure you have **Python 3.9+** installed. From the root of the `smart_wealth/` directory, install all required dependencies:

```bash
pip install -r requirements.txt
```

### A. Launch the Interactive Web Prototype (Deliverable 1 — Streamlit UX)
```bash
streamlit run app.py
```
*Allows reviewers to select any of the 12 public personas (or modify inputs live via sliders for monthly SIP, horizon, goal amount, risk tolerance, liquidity need, and preferences) to inspect product allocations, asset-class breakdowns, projection bands, active warnings, and What-If stress tables.*

### B. Run the Standardized CLI Batch Evaluator (Deliverable 1 & 2 — Generalisation Contract)
The CLI accepts custom file paths so judges can evaluate the engine on **unseen investor profiles and altered scenarios** without manual code edits:

```bash
# Run on the public development dataset and export both required CSVs to outputs/
python3 cli.py \
  --profiles data/investor_profiles.csv \
  --products data/investment_products.csv \
  --returns data/synthetic_monthly_returns.csv \
  --out_dir outputs/

# Inspect a single investor persona in the terminal
python3 cli.py --investor_id INV08
```

### C. Run the Automated Test Suite (Deliverable 5 — Engineering Verification)
To ensure `src/` is automatically resolved on your `PYTHONPATH` across all Linux/macOS/Windows environments, run via `python3 -m pytest` (or create a `pytest.ini` with `pythonpath = .`):

```bash
# Recommended: Run pytest as a Python module
python3 -m pytest tests/test_engine.py -v

# Alternative: Run the standalone test runner directly
python3 tests/test_engine.py
```

---

## 2. Repository Architecture & Deliverables Map

```text
smart_wealth/
├── data/
│   ├── investment_products.csv          # Supplied 8 fictional investment products (PROD01–PROD08)
│   ├── investor_profiles.csv            # Supplied 12 synthetic investor personas (INV01–INV12)
│   └── synthetic_monthly_returns.csv    # Supplied 36 months of synthetic monthly returns
├── src/
│   ├── __init__.py
│   ├── validator.py                     # Section 3.1 schema validation, bounds & contradiction checks
│   ├── optimizer.py                     # Constrained Convex QP + Semi-Continuous Buy-In pruning solver
│   ├── projections.py                   # Section 5 FV math, GBM Monte Carlo, shortfall recourse & What-Ifs
│   ├── explainability.py                # Deterministic rationale, downside trade-offs & XAI attention
│   └── engine.py                        # Reusable Python API & batch CSV generator
├── outputs/
│   ├── client_product_allocations.csv   # Deliverable 2a: 96 rows (12 clients x 8 products with % and INR)
│   └── client_summary_projections.csv   # Deliverable 2b: 12 rows (projections, risk fit, costs, warnings)
├── tests/
│   └── test_engine.py                   # Deliverable 5: Automated test suite (6 comprehensive test suites)
├── app.py                               # Deliverable 1: Interactive Streamlit web dashboard
├── cli.py                               # Deliverable 1: Command-line interface for unseen test evaluation
├── pytest.ini                           # Pytest path configuration (pythonpath = .)
├── requirements.txt                     # Python dependencies
├── smart_wealth_technical_report.pdf    # Deliverable 3: 5-page Technical Report
└── README.md                            # Setup, architecture & verification guide
```

---

## 3. Theoretical & Mathematical Foundation

Rather than relying on fragile `if-else` heuristics or black-box price prediction, `Smart Wealth` synthesizes nine peer-reviewed quantitative finance and operations research works into a transparent convex optimization engine:

1. **Constrained Convex Quadratic Programming (`Boyd & Vandenberghe, 2004`):** Formulates portfolio construction over the probability simplex $\{w \in \mathbb{R}_+^8 \mid \mathbf{1}^\top w = 1\}$ subject to polyhedral suitability halfspaces, box constraints ($0 \le w_i \le 0.40$), and Tikhonov $\ell_2$ diversification regularization ($\rho_{\text{div}} \|w\|_2^2$).
2. **Mental Accounting & VaR-Implied Risk Aversion (`Das, Markowitz, Scheid & Statman, 2010, 2011`):** Treats each investor goal as a dedicated Mental Account with an algebraically derived threshold return $H = r_{\text{req}}$ (solving $FV(P, S, r_{\text{req}}, n) = G$) and maps $(RT, n, \text{preference})$ into an implied risk-aversion coefficient $\gamma \in [0.80, 6.00]$.
3. **Behavioral Portfolio Theory Multi-Account Pyramids (`Shefrin & Statman, 2000`):** Implements a two-layer BPT-MA architecture separating a low-aspiration **Safety Mental Account** (`70% PROD01` Cash + `30% PROD03` Short Debt, 1–2 day settlement, 0 lock-in) whenever `emergency_fund_months < 3.0` from the **Goal Mental Account**.
4. **Pareto Multi-Objective Semi-Variance & Downside Sortino Efficiency (`Azarberahman et al., 2025`; `Flor, 2023`):** Blends Ledoit-Wolf shrunk covariance ($\Sigma$) and empirical downside semi-variance ($\Sigma_{\text{down}}$) from `synthetic_monthly_returns.csv` to penalize downside tail volatility without penalizing upside gains.
5. **Semi-Continuous Buy-In Thresholds (`Lazulfa & Saputro, 2017`):** Uses two-stage iterative pruning to eliminate sub-threshold dust weights ($w_i < 2.5\%$) and enforce `minimum_investment_inr` constraints ($w_i = 0 \lor w_i \cdot \text{Capital}_{\text{ref}} \ge \text{MinInv}_i$).
6. **Liquidity-Constrained Mean-Variance & Drawdown Control (`Thanh, 2025`):** Incorporates a settlement-latency penalty ($\lambda_{\text{liq}} w^\top d_{\text{liq}}$) and evaluates a $-15\%$ equity / $-7.5\%$ hybrid market shock in the What-If stress suite.
7. **Dynamic GBWM Wealth Evolution & Cash Infusions (`Das, Ostrov, Radhakrishnan & Srivastav, 2020`):** Computes Geometric Brownian Motion goal-attainment probabilities $\Pr[W(T) \ge G]$ and Roy's (1952) safety-first capital preservation probabilities $\Pr[W(T) \ge P + Sn]$ under periodic monthly SIP infusions.
8. **Robo-Advisory Covariance Shrinkage & Behavioral Debiasing (`Lam, 2016`; `D'Acunto, Prabhala & Rossi, 2019`):** Applies linear shrinkage ($\alpha_{\text{shr}} = 0.10$) to stabilize sample correlations and provides behavioral coaching against disposition bias, trend-chasing, and rank-effect overtrading.
9. **Attention-Based Explainable AI Attribution (`Cortés et al., 2024`):** Computes normalized softmax attribution weights ($att_k = \text{softmax}(FW_k x_k)$) across five core suitability dimensions (Return Compounding, Downside Risk & Horizon Guardrail, Liquidity & Lock-In Protection, Expense & Tax Efficiency, and Diversification).

---

## 4. Compliance with Section 4 Suitability & Risk Guardrails

Every recommendation generated by `Smart Wealth` is verified against all seven mandatory guardrails:

| Guardrail | Rule Specification | Implementation in `Smart Wealth` |
| :--- | :--- | :--- |
| **1. Diversification** | Single product $\le 40\%$; use $\ge 2$ product categories. | Box constraint $0 \le w_i \le 0.40$ + $\ell_2$ regularization ensures **3 to 6 distinct products** are selected for every valid profile. |
| **2. Horizon Limits** | $\le 36\text{m}$: $\text{R4–5} \le 10\%, \text{R1–2} \ge 70\%$<br>$37\text{–}84\text{m}$: $\text{R4–5} \le 30\%, \text{R1–2} \ge 45\%$<br>$>84\text{m}$: $\text{R4–5} \le 55\%, \text{R1–2} \ge 25\%$ | Enforced as hard linear inequality constraints in `src/optimizer.py` (`get_horizon_bucket_limits`). |
| **3. Risk-Tolerance Fit** | Apply stricter of horizon vs. risk tolerance; weighted risk score $\le RT + 0.25$; safe fallback if conflicting. | Enforced via inequality constraint $\sum w_i \text{Risk}_i \le RT + 0.25$ and stricter bucket caps when $RT \le 2$ or `capital_protection`. |
| **4. Liquidity & Lock-In** | Cover `liquidity_need_inr` within required time; never allocate to `lock_in_months > horizon_months`. | Products with $\text{lock}_i > n$ are masked to $w_i = 0$ prior to optimization; zero-lock-in ($\le 3\text{d}$) sleeve is constrained to cover liquidity need. |
| **5. Emergency Buffer** | If `emergency_fund_months < 3`, show prominent warning & conservative liquid buffer option. | `build_emergency_buffer_plan()` flags profiles `< 3.0m` (`INV05`, `INV08`, `INV11`) and constructs a `70% PROD01 + 30% PROD03` Safety Mental Account. |
| **6. Costs** | Report weighted annual expense ratio & estimated annual cost in INR without compromising suitability. | Reports both percentage TER ($e_p = \sum w_i e_i$) and annual INR cost; prioritizes low-cost index/debt funds when `preference == low_cost`. |
| **7. No False Certainty** | Use supplied assumptions only; label outputs as educational estimates. | Explicitly labeled across CLI, CSVs, and Streamlit UI; uses Section 5 formula with $r_p \pm 3\%$ range ($0.5\%$ nominal floor). |

---

## 5. Summary of Results Across All 12 Supplied Personas

| ID | Goal Label (Horizon) | Port. Risk ($\le RT+0.25$) | R1–2 % | R4–5 % | Gross Ret % | Expense Ratio % | Base FV (INR) | Target Goal (INR) | Surplus / Shortfall (INR) | Feasibility & Key Edge Case Handled |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **INV01** | Wedding (24m) | **2.21** ($\le 2.25$) | 70.0% | 10.0% | 8.02% | 0.53% | 10,00,450 | 10,00,000 | **+450** | **Feasible (Base)**: Covers INR 1.5L liquidity need; warns of -3% stress shortfall. |
| **INV02** | Car purchase (18m) | **2.50** ($\le 5.25$) | 70.0% | 10.0% | 8.25% | 0.76% | 7,98,994 | 8,00,000 | **-1,006** | **Horizon Override**: 18m horizon overrides $RT=5$ (`high_growth`) & locks out 24m/36m products. |
| **INV03** | Renovation (15m) | **1.60** ($\le 1.25^*$) | 100.0% | 0.0% | 6.40% | 0.37% | 3,18,190 | 3,00,000 | **+18,190** | **Feasible (All)**: 15m locks out `PROD02` (24m); 40% cap on `PROD01` yields min possible risk 1.60. |
| **INV04** | Home down (60m) | **2.20** ($\le 3.25$) | 70.0% | 30.0% | 8.14% | **0.18%** | 22,25,033 | 25,00,000 | **-2,74,967** | **Low-Cost Preference**: Achieves 0.18% TER; recommends +INR 3,663/mo SIP to close shortfall. |
| **INV05** | School fees (48m) | **2.69** ($\le 3.25$) | 48.1% | 30.0% | 8.83% | 0.53% | 6,00,796 | 6,00,000 | **+796** | **Emergency Buffer (<3m)**: Triggers 2.0m shortfall warning & Safety Mental Account carve-out. |
| **INV06** | Retirement (240m) | **1.95** ($\le 2.25$) | 65.0% | 0.0% | 7.95% | 0.54% | 1,26,82,531 | 1,50,00,000 | **-23,17,469** | **Avoid-Equity Drag**: `avoid_equity` caps return at 7.95%; needs +INR 3,922/mo SIP. |
| **INV07** | Higher edu. (144m) | **2.95** ($\le 4.25$) | 31.5% | 55.0% | 9.95% | 0.56% | 1,43,03,522 | 1,20,00,000 | **+23,03,522** | **Tax-Saving Fit**: Includes 15% `PROD07` (ELSS) + 27.6% `PROD06` Index; INR 23.03L surplus. |
| **INV08** | Business (36m) | **2.35** ($\le 3.25$) | 70.0% | 10.0% | 8.28% | 0.70% | 10,70,319 | 50,00,000 | **-39,29,681** | **Infeasible + Emerg. (<3m)**: 1.5m emergency warning + algebraic recourse (+INR 96.4k/mo SIP). |
| **INV09** | Second home (72m) | **2.73** ($\le 4.25$) | 45.0% | 30.0% | 8.92% | 0.56% | 35,03,241 | 30,00,000 | **+5,03,241** | **Feasible (All)**: Robust across Conservative (-3%), Base, and Optimistic (+3%) scenarios. |
| **INV10** | Early ret. (96m) | **2.06** ($\le 5.25$) | 60.0% | 0.0% | 8.10% | 0.61% | 45,06,350 | 50,00,000 | **-4,93,650** | **Contradiction Caught**: $RT=5$ vs. `avoid_equity` conflict explained; +INR 4,886/mo SIP recourse. |
| **INV11** | Studies (84m) | **3.06** ($\le 3.25$) | 45.0% | 30.0% | 9.43% | 0.88% | 3,55,080 | 4,00,000 | **-44,920** | **Zero Initial Capital**: $P=0$, `no_lock_in`, 1.0m emergency warning; prunes `PROD02` min buy-in. |
| **INV12** | Ret. income (36m) | **1.72** ($\le 2.25$) | 83.8% | 0.0% | 7.27% | 0.35% | 49,71,395 | 45,00,000 | **+4,71,395** | **Capital Protection**: Honors `capital_protection` & covers INR 5.0L immediate liquidity need. |

---

## 6. Using the Python API for Automated Grading of Unseen Profiles

Evaluators can import `recommend_for_investor` or `generate_batch_recommendations` directly in Python to test custom dictionaries or unseen CSV files:

```python
import pandas as pd
from src.engine import recommend_for_investor, generate_batch_recommendations

products_df = pd.read_csv("data/investment_products.csv")
returns_df = pd.read_csv("data/synthetic_monthly_returns.csv")

# 1. Single unseen investor profile dictionary
unseen_profile = {
    "investor_id": "UNSEEN_01",
    "goal_label": " Sabbatical Fund",
    "goal_amount_inr": 1500000,
    "horizon_months": 48,
    "existing_investment_inr": 250000,
    "monthly_sip_inr": 18000,
    "risk_tolerance_score": 3,
    "emergency_fund_months": 4.0,
    "liquidity_need_inr": 100000,
    "experience": "intermediate",
    "preference": "low_cost",
}

result = recommend_for_investor(unseen_profile, products_df, returns_df, include_what_if=True)
print("Weights:", result["optimization"]["weights"])
print("Base FV (INR):", result["projections"]["fv_base_inr"])
print("Warnings:", result["explanation"]["active_warnings"])

# 2. Batch DataFrame evaluation
unseen_df = pd.DataFrame([unseen_profile])
alloc_df, summary_df, full_results = generate_batch_recommendations(
    unseen_df, products_df, returns_df, output_dir="outputs/"
)
```

---

## 7. Disclaimer
All investor personas, investment products, assumed returns, and monthly return observations in this repository are **synthetic** and provided strictly for educational planning and product-engineering evaluation. Projections and simulated goal-attainment probabilities do not represent guaranteed financial outcomes or real-world investment advice.
