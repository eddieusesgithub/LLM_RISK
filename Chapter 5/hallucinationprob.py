import pandas as pd
from fractions import Fraction
from statsmodels.miscmodels.ordinal_model import OrderedModel

CSV_PATH = "dataset_sc_mb.csv"

df = pd.read_csv(CSV_PATH)
#rename columns
rename_map = {
    "Hallucination Score": "halluc",
    "METEOR_Score": "meteor",
    "BERTScore_F1": "bert_f1",
}
for k, v in rename_map.items():
    if k in df.columns:
        df = df.rename(columns={k: v})

# parsing hallucination labels that are like strings from excel spreadsheet "1/3", "2/3", "0", "1" 
def parse_halluc(x):
    if pd.isna(x):
        raise ValueError("NaN hallucination label encountered.")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
  
    if s in {"1/3", "⅓", "1⁄3"}:
        return 1.0/3.0
    if s in {"2/3", "⅔", "2⁄3"}:
        return 2.0/3.0
    if s in {"0", "0.0"}:
        return 0.0
    if s in {"1", "1.0"}:
        return 1.0

# convert to numeric in [0, 1/3, 2/3, 1]
halluc_float = df["halluc"].apply(parse_halluc).astype(float)

# map {0, 1/3, 2/3, 1} -> {1,2,3,4} just for easier labelling
# 0 -> 0 -> +1 => 1 (Hallucinatory)
# 1/3 -> 1 -> +1 => 2 (Bad)
# 2/3 -> 2 -> +1 => 3 (Good)
# 1 -> 3 -> +1 => 4 (Perfect)
y = (halluc_float.mul(3).round().clip(0, 3).astype(int) + 1).astype(int)

# predictors meteor and bertscore
X = df[["meteor", "bert_f1"]].astype(float)

# fit ordinal logistic regression (proportional odds, logit link)
model = OrderedModel(y, X, distr="logit")
res = model.fit(method="bfgs", disp=False)

# predict class probabilities (n x 4)
probs = res.model.predict(res.params, exog=X)
if not isinstance(probs, pd.DataFrame):
    probs = pd.DataFrame(probs, columns=["P_Hallucinatory", "P_Bad", "P_Good", "P_Perfect"])
else:
    probs.columns = ["P_Hallucinatory", "P_Bad", "P_Good", "P_Perfect"]

# combine with original inputs + original label text (keep original columns too)
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
