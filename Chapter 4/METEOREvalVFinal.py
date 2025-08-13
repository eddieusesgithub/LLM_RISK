import csv
import evaluate
import time

INPUT_CSV_FILENAME = 'prompts_generated_sc.csv' # dataset file
OUTPUT_CSV_FILENAME = 'dataset_sc.csv' # save to this file


# load METEOR from evaluate library from hugging face
meteor = evaluate.load('meteor')

#main algorithm
def main():
    all_data = []
    
    print(f"Opened '{INPUT_CSV_FILENAME}'.")
    
    with open(INPUT_CSV_FILENAME, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            input_headers = next(reader) # read headings
            
            ideal_response_col_index = input_headers.index('Ideal Response') # get ground truths
            generated_response_col_index = input_headers.index('LLM Response') # get llm responses

            all_data.append(input_headers + ['METEOR_Score']) # add new columns for each metric

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

                #compute METEOR
                meteor_results = meteor.compute(
                    predictions=[generated_response],
                    references=[ideal_response]
                )
                meteor_score = meteor_results['meteor']

                # appendd scores to the current row
                new_row = row + [meteor_score]
                all_data.append(new_row)

                # write progress to output file evrey 10 rows 
                if (i + 1) % 10 == 0 or (i + 1) == total_prompts: # corrected total count for last save
                    with open(OUTPUT_CSV_FILENAME, mode='w', newline='', encoding='utf-8') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(all_data)
                    print(f"Progress saved for {i+1}/{total_prompts} rows.")
                
                time.sleep(0.05) 

    print(f"\nEvaluated {len(all_data) - 1} total responses.")

    print(f"\nAll METEOR scores computed and saved to '{OUTPUT_CSV_FILENAME}'.")

if __name__ == "__main__":
    main()

