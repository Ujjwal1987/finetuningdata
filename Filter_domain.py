import pandas as pd
import os

# --- Assumptions ---
# Assuming you successfully loaded and merged your data into this DataFrame:
# (You might need to reload it if the script execution was interrupted)
try:
    # Assuming the latest merged file was saved here
    merged_file_path = os.path.join("Data", "g_claims_2016.csv")

    # Reload the data (adjust engine/separator if needed based on previous errors)
    df_merged_with_cpc_group = pd.read_csv(merged_file_path)

    # 1. Define the filter criteria
    filter_strings = ['G06K7', 'B41J3']  # Added B41J3 filter

    # 2. Filter the DataFrame for both criteria
    # Create a combined condition using OR logic
    mask = df_merged_with_cpc_group['cpc_group'].str.contains(filter_strings[0], na=False, case=False)

    # Add the second filter condition
    for filter_string in filter_strings[1:]:
        mask = mask | df_merged_with_cpc_group['cpc_group'].str.contains(filter_string, na=False, case=False)

    df_filtered_combined = df_merged_with_cpc_group[mask].copy()

    # Convert 'dependent' to a string type for robust checking, coercing errors to NaN
    # df_filtered_combined['dependent'] = pd.to_numeric(df_filtered_combined['dependent'], errors='coerce')

    # Condition to keep rows where 'dependent' is NaN (or 0, for completeness)
    condition_independent = df_filtered_combined['dependent'].isna() | (df_filtered_combined['dependent'] == "")

    df_independent = df_filtered_combined[condition_independent]
    print(df_independent.columns.tolist())

    # 3. Merge Rows: Combine 'claim_text' based on 'patent_id'
    # We use groupby() and agg() to concatenate the claim texts.
    df_merged_text = df_independent.groupby('patent_id').agg(
        merged_claim_text=('claim_text', lambda x: ' [SEP] '.join(x.astype(str))),
        cpc_group=('cpc_group', 'first'),
        dependent=('dependent', 'first')
    ).reset_index()

    # 4. Display and Save the Result
    output_path = os.path.join("Data", "g_patents_merged_text_2016.csv")
    df_merged_text.to_csv(output_path, index=False)

    print(
        f"\nSuccessfully merged claim texts for both G06K7 and B41J3. Resulting DataFrame has {len(df_merged_text)} patents.")
    print("Merged Data Structure (first 3 rows):")
    print(df_merged_text.head(3))

    # Optional: Display summary statistics
    print(f"\nCPC Group distribution in filtered data:")
    print(df_independent['cpc_group'].value_counts(dropna=False))

except FileNotFoundError:
    print(
        f"Error: The merged file at '{merged_file_path}' was not found. Please ensure the merge step was completed and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")