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

    # 1. Define the filter criterion
    filter_string = 'B41J3'

    # 2. Filter the DataFrame
    # Use str.contains() to check if the 'cpc_group' column contains the filter_string.
    # The 'na=False' ensures that any NaN values in 'cpc_group' are treated as non-matching (False).
    df_filtered_b41j3 = df_merged_with_cpc_group[
        df_merged_with_cpc_group['cpc_group'].str.contains(filter_string, na=False)
    ].copy()  # .copy() is good practice after filtering

    # Convert 'dependent' to a string type for robust checking, coercing errors to NaN
    # df_filtered_b41j3['dependent'] = pd.to_numeric(df_filtered_b41j3['dependent'], errors='coerce')

    # Condition to keep rows where 'dependent' is NaN (or 0, for completeness)


    condition_independent = df_filtered_b41j3['dependent'].isna() | (df_filtered_b41j3['dependent'] == "")

    df_independent = df_filtered_b41j3[condition_independent]
    print(df_independent.columns.tolist())

#
#     # print(f"\nFiltered out dependent claims. Remaining rows (Independent Claims): {len(df_independent)}")
#
#     # 2. Merge Rows: Combine 'claim_text' based on 'patent_id'
#     # We use groupby() and agg() to concatenate the claim texts.
#
#     # The separator ' [SEP] ' is added between claim texts for clear demarcation.
#     # We'll work with the original 'df' to merge ALL claim texts, as usually done for document analysis.
#     # If you only want to merge *independent* claim texts, use 'df_independent' here.
#     # Assuming you want ALL claim texts for document analysis:
#
    df_merged_text = df_independent.groupby('patent_id').agg(
#         # Aggregate claim_text by joining all claim strings with a separator
        merged_claim_text=('claim_text', lambda x: ' [SEP] '.join(x.astype(str))),
#         # Keep the first cpc_group for the patent (since all claims for a patent should share the same main CPC)
        cpc_group=('cpc_group', 'first'),
        dependent=('dependent', 'first')
    ).reset_index()

    # print(df_merged_text['dependent'].value_counts(dropna=False))
    # print(df_merged_text.columns.tolist())

#
#     # 3. Display and Save the Result
#     print(f"\nSuccessfully merged claim texts. Resulting DataFrame has {len(df_merged_text)} patents.")
#     print("Merged Data Structure (first 3 rows):")
#     print(df_merged_text.head(3))
#
    output_path = os.path.join("Data", "g_patents_printer_merged_text.csv")
    df_merged_text.to_csv(output_path, index=False)
#     print(f"\nMerged patent-level data saved to: {output_path}")
#
#     # # 3. Display the results
#     # print(f"Original number of rows: {len(df_merged_with_cpc_group)}")
#     # print(f"Number of rows filtered for '{filter_string}': {len(df_filtered_b41j3)}")
#     #
#     # # 4. Optional: Display the first few rows of the filtered data
#     # print("\nFirst 5 rows of the filtered data:")
#     # print(df_filtered_b41j3[['patent_id', 'cpc_group']].head())
#     #
#     # # 5. Optional: Save the filtered data
#     # output_path = os.path.join("Data", "g_claims_cpc_printer.csv")
#     # df_filtered_b41j3.to_csv(output_path, index=False)
#     # print(f"\nFiltered data saved to: {output_path}")

except FileNotFoundError:
    print(
        f"Error: The merged file at '{merged_file_path}' was not found. Please ensure the merge step was completed and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")