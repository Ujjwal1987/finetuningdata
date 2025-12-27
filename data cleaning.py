import numpy as np
import pandas as pd
import os

# Define the path to the file within the "Data" folder
file_path = os.path.join("Data", "g_claims_2019.tsv")
cpc_file_path = os.path.join("Data", "g_cpc_current.tsv")

try:
    # Read the TSV file with memory optimization and explicit dtypes
    print("Reading TSV file...")
    df = pd.read_csv(file_path, sep='\t', low_memory=False)

    # Get the column names from the DataFrame's columns attribute
    column_names = df.columns.tolist()
    print("Column Names:")
    for name in column_names:
        print(name)

    # Filter the DataFrame to remove rows where 'patent_id' starts with specified prefixes
    prefixes_to_remove = ['D', 'PP', 'RE']

    # Create a more robust filtering approach that handles NaN values properly
    condition_prefixes = ~df['patent_id'].str.startswith(tuple(prefixes_to_remove),
                                                         na=False) if 'patent_id' in df.columns else pd.Series(
        [True] * len(df))
    condition_blanks = pd.notna(
        df['patent_id'].replace(r'^\s*$', np.nan, regex=True)) if 'patent_id' in df.columns else pd.Series(
        [True] * len(df))

    # Apply filters only if patent_id column exists
    if 'patent_id' in df.columns:
        filtered_df = df[condition_prefixes & condition_blanks]
        print(f"Number of rows after filtering: {len(filtered_df)}")
    else:
        filtered_df = df
        print("No 'patent_id' column found - keeping all rows")

    # Handle CPC data merging with better error handling
    if os.path.exists(cpc_file_path):
        print("Reading CPC file...")
        df_cpc = pd.read_csv(cpc_file_path, sep='\t', low_memory=False)

        # Check if required columns exist
        if 'patent_id' in df_cpc.columns and 'cpc_group' in df_cpc.columns:
            # Ensure consistent data types for patent_id before merging
            print("Checking data types...")
            print(f"Original df_cpc patent_id dtype: {df_cpc['patent_id'].dtype}")
            print(f"Original filtered_df patent_id dtype: {filtered_df['patent_id'].dtype}")

            # Convert both patent_id columns to string type to ensure compatibility
            df_cpc['patent_id'] = df_cpc['patent_id'].astype(str)
            filtered_df['patent_id'] = filtered_df['patent_id'].astype(str)

            print(f"After conversion - df_cpc patent_id dtype: {df_cpc['patent_id'].dtype}")
            print(f"After conversion - filtered_df patent_id dtype: {filtered_df['patent_id'].dtype}")

            # Create a copy to avoid modifying original data
            df_cpc_copy = df_cpc.copy()
            filtered_df_copy = filtered_df.copy()

            # Merge the dataframes ensuring consistent types
            df_merged_with_cpc_group = pd.merge(
                filtered_df_copy,
                df_cpc_copy[['patent_id', 'cpc_group']],  # Only select needed columns
                on='patent_id',
                how='left',
                suffixes=('', '_right')  # Avoid column name conflicts
            )

            print(f"Merged data shape: {df_merged_with_cpc_group.shape}")

        else:
            print("Warning: Required columns 'patent_id' and 'cpc_group' not found in CPC file")
            df_merged_with_cpc_group = filtered_df
    else:
        print("Warning: CPC file not found - proceeding with original filtered data")
        df_merged_with_cpc_group = filtered_df

    # Delete rows where "dependent" column is NOT null/nan (keep only rows where "dependent" is null/nan)
    if 'dependent' in df_merged_with_cpc_group.columns:
        # Keep rows where "dependent" column is null/NaN
        df_final = df_merged_with_cpc_group[df_merged_with_cpc_group['dependent'].isna()]
        print(f"Final dataset shape after removing non-null 'dependent' rows: {df_final.shape}")

        # Show how many rows were removed
        removed_rows = len(df_merged_with_cpc_group) - len(df_final)
        print(f"Removed {removed_rows} rows where 'dependent' column had values")
    else:
        print("Warning: 'dependent' column not found in the merged data")
        df_final = df_merged_with_cpc_group

    # Remove duplicate claim_sequence values for each patent_id, keeping only the first occurrence
    if 'patent_id' in df_final.columns and 'claim_sequence' in df_final.columns:
        print("Removing duplicate claim_sequence values for each patent_id...")

        # Sort by patent_id and claim_sequence to ensure proper ordering
        df_final_sorted = df_final.sort_values(['patent_id', 'claim_sequence'])

        # Keep only first occurrence of each (patent_id, claim_sequence) combination
        df_result = df_final_sorted.drop_duplicates(subset=['patent_id', 'claim_sequence'], keep='first')

        print(f"Resulting dataset shape after removing duplicate claim_sequences: {df_result.shape}")

        # Show how many rows were removed
        removed_rows = len(df_final_sorted) - len(df_result)
        if removed_rows > 0:
            print(f"Removed {removed_rows} duplicate rows")

        # Re-sort by original order (if needed)
        df_final = df_result.sort_values(['patent_id', 'claim_sequence'])
    else:
        print("Warning: 'patent_id' or 'claim_sequence' column not found in the data")
        print("Columns available:", df_final.columns.tolist())

    # Save the filtered DataFrame to a new CSV file
    output_path = os.path.join("Data", "g_claims_2019.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Successfully saved filtered data to: {output_path}")

except KeyError as e:
    print(f"Error: A required column was not found - {e}")
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()
