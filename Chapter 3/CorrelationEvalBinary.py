import pandas as pd
from scipy.stats import spearmanr

# load dataset (binary)
INPUT_CSV_FILENAME = 'dataset_binary_evaluated.csv'

# main script
def main():
 
    df = pd.read_csv(INPUT_CSV_FILENAME)
    print(f"Data loaded from '{INPUT_CSV_FILENAME}'") 
    # get ground truths 
    human_labels_col = 'Hallucination Score'
    # define the columns for automated metric scores
    automated_metrics_cols = [
        'ROUGE_L_Score',
        'BLEU_Score',
        'METEOR_Score',
        'BERTScore_F1'
    ]

    # convert human labels and metric scores to numeric, coercing errors to NaN
    df[human_labels_col] = pd.to_numeric(df[human_labels_col], errors='coerce')
    for col in automated_metrics_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows where any of the relevant columns are NaN (e.g., if a score couldn't be computed)
    df_cleaned = df.dropna(subset=[human_labels_col] + automated_metrics_cols)

    # compute correlations
    correlation_results = {}
    for metric_col in automated_metrics_cols:
        # return spearmans coefficient + p value
        corr, p_value = spearmanr(df_cleaned[human_labels_col], df_cleaned[metric_col])
        correlation_results[metric_col] = {'correlation': corr, 'p_value': p_value}
        
        print(f"  - {metric_col}: Spearman's rho = {corr:.4f}, p-value = {p_value:.4f}")

    # determine the best metric based on highest absolute correlation
    best_metric = None
    highest_corr_abs = -1.0
    
    for metric, results in correlation_results.items():
        if results['correlation'] > highest_corr_abs:
            highest_corr_abs = results['correlation']
            best_metric = metric
    print(f"\nThe most efficacious metric is: {best_metric} (Rho = {highest_corr_abs:.4f})")
if __name__ == "__main__":
    main()

