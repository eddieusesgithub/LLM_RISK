import csv
import evaluate
import time

INPUT_CSV_FILENAME = 'dataset_ordinal.csv' # dataset file
OUTPUT_CSV_FILENAME = 'dataset_ordinal_evaluated.csv' # save to this file


# load metrics from evaluate library from hugging face
print("Loading evaluation metrics...")
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')
print("Metrics loaded successfully.")

#main algorithm
def main():
    all_data = []
    
    print(f"Opened '{INPUT_CSV_FILENAME}'.")
    
    with open(INPUT_CSV_FILENAME, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            input_headers = next(reader) # read headings
            
            ideal_response_col_index = input_headers.index('Ideal Response') # get ground truths
            generated_response_col_index = input_headers.index('LLM Response') # get llm responses

            all_data.append(input_headers + ['ROUGE_L_Score', 'BLEU_Score', 'METEOR_Score', 'BERTScore_F1']) # add new columns for each metric

            rows_from_csv = list(reader) #gtting dimensions for the dataset
            total_prompts = len(rows_from_csv)

            for i, row in enumerate(rows_from_csv): # iterate through the list of rows
                if not row: # skip empty rows
                    continue

                ideal_response = row[ideal_response_col_index]
                generated_response = row[generated_response_col_index]

                # make sure each responses are strings; handle potential empty strings
                ideal_response = str(ideal_response).strip() if ideal_response is not None else ""
                generated_response = str(generated_response).strip() if generated_response is not None else ""

                if not generated_response: # skip if generated response is empty
                    print(f"Warning: Skipping row {i+2} due to empty 'Generated Response'.")
                    all_data.append(row + ['', '', '', '']) # add empty scores to initialise
                    continue

                print(f"Evaluating row {i+1}/{total_prompts}: Prompt '{row[0][:50]}...'") #print each row so we can see progress

                #compute ROUGE
                #ROUGE requires lists of strings for references and predictions
                rouge_results = rouge.compute(
                    predictions=[generated_response],
                    references=[ideal_response]
                )
                rouge_l = rouge_results['rougeL'] # we chose ROUGE-L as outlined in the main report

                #compute BLEU
                #BLEU expects a list of predictions and a list of lists of references
                bleu_results = bleu.compute(
                    predictions=[generated_response],
                    references=[[ideal_response]] # note the nested list for references
                )
                bleu_score = bleu_results['bleu']

                #compute METEOR
                meteor_results = meteor.compute(
                    predictions=[generated_response],
                    references=[ideal_response]
                )
                meteor_score = meteor_results['meteor']

                #compute BERTScore
                #BERTScore returns precision, recall, and F1. We'll use F1 here.

                bertscore_results = bertscore.compute(
                    predictions=[generated_response],
                    references=[ideal_response],
                    lang="en" # specify english for the BERTScore model embeddings
                )
                bertscore_f1 = bertscore_results['f1'][0] # F1 score for the first (and only) prediction

                # appendd scores to the current row
                new_row = row + [rouge_l, bleu_score, meteor_score, bertscore_f1]
                all_data.append(new_row)

                # write progress to output file evrey 10 rows 
                if (i + 1) % 10 == 0 or (i + 1) == total_prompts: # corrected total count for last save
                    with open(OUTPUT_CSV_FILENAME, mode='w', newline='', encoding='utf-8') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(all_data)
                    print(f"Progress saved for {i+1}/{total_prompts} rows.")
                
                time.sleep(0.05) 

    print(f"\nEvaluated {len(all_data) - 1} total responses.")

    print(f"\nAll metric scores computed and saved to '{OUTPUT_CSV_FILENAME}'.")

if __name__ == "__main__":
    main()

