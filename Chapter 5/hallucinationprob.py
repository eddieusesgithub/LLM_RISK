import pandas as pd
from fractions import Fraction
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ====== USER: change this path for each file ======
CSV_PATH = "dataset_sc_mb.csv"
# ==================================================

df = pd.read_csv(CSV_PATH)

# Tolerant column rename (handles a few naming variants)
rename_map = {
    "Hallucination Score": "halluc",
    "METEOR_Score": "meteor",
    "BERTScore_F1": "bert_f1",
}
for k, v in rename_map.items():
    if k in df.columns:
        df = df.rename(columns={k: v})

required = {"halluc", "meteor", "bert_f1"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

# --- Parse hallucination labels that may be strings like "1/3", "2/3", "0", "1" ---
def parse_halluc(x):
    if pd.isna(x):
        raise ValueError("NaN hallucination label encountered.")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    # common textual forms
    if s in {"1/3", "⅓", "1⁄3"}:
        return 1.0/3.0
    if s in {"2/3", "⅔", "2⁄3"}:
        return 2.0/3.0
    if s in {"0", "0.0"}:
        return 0.0
    if s in {"1", "1.0"}:
        return 1.0
    # last resort: try float or Fraction
    try:
        return float(s)
    except Exception:
        try:
            return float(Fraction(s))
        except Exception:
            raise ValueError(f"Unrecognized hallucination label: {x}")

# Convert to numeric in [0, 1/3, 2/3, 1]
halluc_float = df["halluc"].apply(parse_halluc).astype(float)

# Map {0, 1/3, 2/3, 1} -> {1,2,3,4} robustly by multiplying by 3 and rounding
# 0 -> 0 -> +1 => 1 (Hallucinatory)
# 1/3 -> 1 -> +1 => 2 (Bad)
# 2/3 -> 2 -> +1 => 3 (Good)
# 1 -> 3 -> +1 => 4 (Perfect)
y = (halluc_float.mul(3).round().clip(0, 3).astype(int) + 1).astype(int)

# Predictors
X = df[["meteor", "bert_f1"]].astype(float)

# Fit ordinal logistic regression (proportional odds, logit link)
model = OrderedModel(y, X, distr="logit")
res = model.fit(method="bfgs", disp=False)

# Predict class probabilities (n x 4)
probs = res.model.predict(res.params, exog=X)
if not isinstance(probs, pd.DataFrame):
    probs = pd.DataFrame(probs, columns=["P_Hallucinatory", "P_Bad", "P_Good", "P_Perfect"])
else:
    probs.columns = ["P_Hallucinatory", "P_Bad", "P_Good", "P_Perfect"]

# Combine with original inputs + original label text (keep your original column too)
out_df = pd.concat(
    [
        df[["meteor", "bert_f1"]].reset_index(drop=True),
        df[["halluc"]].reset_index(drop=True),  # original label as given
        probs.reset_index(drop=True),
    ],
    axis=1,
)

out_csv = CSV_PATH.replace(".csv", "_probs.csv")
out_df.to_csv(out_csv, index=False)
print(f"Saved probabilities with original scores to {out_csv}")
