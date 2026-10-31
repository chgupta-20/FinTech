# Smart Wealth: Goal-Based Investment Strategy Prototype
**Inter-IIT Tech Meet 15.0 Practice Problem Statement Deliverable**

## 1. Quick Start & Launch Instructions
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate batch recommendation CSVs (CLI / Generalisation Contract)
python cli.py --profiles data/investor_profiles.csv --products data/investment_products.csv --returns data/synthetic_monthly_returns.csv --out_dir outputs/

# 3. Launch Interactive Streamlit Prototype
streamlit run app.py

# 4. Run Automated Pytest Verification Suite
pytest tests/test_engine.py -v