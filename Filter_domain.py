import pandas as pd
import os

# --- Assumptions ---
# Assuming you successfully loaded and merged your data into this DataFrame:
# (You might need to reload it if the script execution was interrupted)
try:
    # Assuming the latest merged file was saved here
    merged_file_path = os.path.join("Data", "g_claims_2019.csv")

    # Reload the data (adjust engine/separator if needed based on previous errors)
    df_merged_with_cpc_group = pd.read_csv(merged_file_path)

    # 1. Define the filter criteria
    filter_strings = []  # Add your actual filter criteria here

    # 2. Filter the DataFrame for both criteria
    # Check if filter criteria is empty
    if len(filter_strings) == 0:
        # If no filter criteria, process all data without filtering
        df_filtered_combined = df_merged_with_cpc_group.copy()
        print("No filter criteria provided. Processing all data.")
        print(f"Column names in original data: {df_merged_with_cpc_group.columns.tolist()}")
    else:
        # Create a combined condition using OR logic
        mask = df_merged_with_cpc_group['cpc_group'].str.contains(filter_strings[0], na=False, case=False)

        # Add the second filter condition (only if there are more than 1 filter strings)
        if len(filter_strings) > 1:
            for filter_string in filter_strings[1:]:
                mask = mask | df_merged_with_cpc_group['cpc_group'].str.contains(filter_string, na=False, case=False)

        df_filtered_combined = df_merged_with_cpc_group[mask].copy()
        print(f"Column names in filtered data: {df_filtered_combined.columns.tolist()}")

    # Convert 'dependent' to a string type for robust checking, coercing errors to NaN
    # df_filtered_combined['dependent'] = pd.to_numeric(df_filtered_combined['dependent'], errors='coerce')

    # Condition to keep rows where 'dependent' is NaN (or empty string)
    condition_independent = df_filtered_combined['dependent'].isna() | (df_filtered_combined['dependent'] == "")

    # Additional safety check to ensure 'dependent' column exists
    if 'dependent' not in df_filtered_combined.columns:
        print("Warning: 'dependent' column not found in filtered data")
        df_independent = df_filtered_combined.copy()
    else:
        df_independent = df_filtered_combined[condition_independent]

    print(f"Column names in independent data: {df_independent.columns.tolist()}")

    # 3. Merge Rows: Combine 'claim_text' based on 'patent_id'
    # We use groupby() and agg() to concatenate the claim texts.
    if 'patent_id' not in df_independent.columns:
        print("Error: 'patent_id' column not found in filtered data")
        raise ValueError("Missing required 'patent_id' column")

    # Ensure we have data to process
    if len(df_independent) == 0:
        print("Warning: No data to process after filtering")
        df_merged_text = pd.DataFrame()
    else:
        df_merged_text = df_independent.groupby('patent_id').agg(
            merged_claim_text=('claim_text', lambda x: ' [SEP] '.join(x.astype(str))),
            cpc_group=('cpc_group', 'first'),
            dependent=('dependent', 'first')
        ).reset_index()

    # 4. Delete 'cpc_group' and 'dependent' columns from final output
    if len(df_merged_text) > 0:
        # Check if the columns exist before trying to drop them
        columns_to_drop = []
        if 'cpc_group' in df_merged_text.columns:
            columns_to_drop.append('cpc_group')
        if 'dependent' in df_merged_text.columns:
            columns_to_drop.append('dependent')

        if columns_to_drop:
            df_final_output = df_merged_text.drop(columns=columns_to_drop, errors='ignore')
            print(f"Deleted columns: {columns_to_drop}")
        else:
            df_final_output = df_merged_text
            print("No 'cpc_group' or 'dependent' columns found to delete")

        # Print final column names
        print(f"Final output columns (after deletion): {df_final_output.columns.tolist()}")

        # Save the final output
        output_path = os.path.join("Data", "g_patents_merged_text.csv")
        df_final_output.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved final output with {len(df_final_output)} rows.")
        print("Final output columns:", df_final_output.columns.tolist())

        # Show first 3 rows of final output
        print("\nFirst 3 rows of final output:")
        print(df_final_output.head(3))

    else:
        print("Warning: No data to save - empty result after filtering")
        # Save empty DataFrame with correct structure
        pd.DataFrame(columns=['patent_id', 'merged_claim_text']).to_csv(
            os.path.join("Data", "g_patents_merged_text_2016.csv"), index=False)
        print(f"Saved empty file with columns: ['patent_id', 'merged_claim_text']")

except FileNotFoundError:
    print(
        f"Error: The merged file at '{merged_file_path}' was not found. Please ensure the merge step was completed and the file exists.")
except IndexError as e:
    print(f"Index error occurred: {e}")
    print("This typically means you're trying to access an index that doesn't exist in a list or array")
    print("Check that filter_strings has elements before accessing them")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(f"Error type: {type(e).__name__}")

