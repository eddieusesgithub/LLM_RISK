import pandas as pd
from scipy.stats import spearmanr

# load dataset (with ordinal labels)
INPUT_CSV_FILENAME = 'dataset_ordinal_evaluated.csv' 

# main script
def main():
    
    df = pd.read_csv(INPUT_CSV_FILENAME)
    print(f"Data loaded from '{INPUT_CSV_FILENAME}'") 

    # define the column for human labels
    human_labels_col = 'Hallucination Score'

    # define the columns for automated metric scores
    automated_metrics_cols = [
        'ROUGE_L_Score',
        'BLEU_Score',
        'METEOR_Score',
        'BERTScore_F1'
    ]

    # need to convert 1/3 and 2/3 into strings 
    # convert the column to string type first to deal with floats
    df[human_labels_col] = df[human_labels_col].astype(str).str.strip()

    # mapping the differennt representations of 1/3 2/3 as strings 
    ordinal_value_map = {
        '0': 0.0,
        '0.0': 0.0,
        '1': 1.0,
        '1.0': 1.0,
        '1/3': 1/3,  # added literal fraction string
        '2/3': 2/3,  
        # used excel to process the data and labelling so this is the exact values used 
        '0.3333333333333333': 1/3, 
        '0.6666666666666666': 2/3,
        '0.33': 1/3,
        '0.67': 2/3,
        '0.333': 1/3,
        '0.667': 2/3,
        # ensure any empty strings  become NaN
        '': float('nan')
    }

    # apply the above mapping. Values not in the map will become NaN
    df['mapped_human_labels'] = df[human_labels_col].map(ordinal_value_map)

    # now use the new mapped column for correlation 
    human_labels_col_for_corr = 'mapped_human_labels'
    required_cols_for_dropna = [human_labels_col_for_corr] + automated_metrics_cols

    # convert automated metric scores to numeric
    for col in automated_metrics_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #  report any rows with NaN values before dropping 
    df_cleaned = df.dropna(subset=required_cols_for_dropna)

    # compute correlations
    correlation_results = {}
    for metric_col in automated_metrics_cols:
        corr, p_value = spearmanr(df_cleaned[human_labels_col_for_corr], df_cleaned[metric_col])
        correlation_results[metric_col] = {'correlation': corr, 'p_value': p_value}
        
        print(f"  - {metric_col}: Spearman's rho = {corr:.4f}, p-value = {p_value:.4f}")

    best_metric = None
    highest_corr_abs = -1.0

    # we want the strongest positive correlation, as higher human score (1.0) means perfect (no hallucination)
     # and higher metric score means better alignment.
    for metric, results in correlation_results.items():
        if results['correlation'] > highest_corr_abs:
            highest_corr_abs = results['correlation']
            best_metric = metric
    print(f"\nThe most efficacious metric is: {best_metric} (Rho = {highest_corr_abs:.4f})")
    
if __name__ == "__main__":
    main()

