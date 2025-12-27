import pandas as pd
import os
from pathlib import Path


def memory_efficient_merge():
    # Define file paths
    merged_desc_path = os.path.join("Data", "g_brf_sum_text_2019.tsv")
    df_patent_loc = os.path.join("Data", "g_patents_merged_text.csv")
    output_path = os.path.join("Data", "g_patents_scanner_printer_merged_text.csv")

    # Read patent data with memory optimization
    print("Reading patent data...")
    # Add low_memory=False to handle mixed types properly
    df_patent = pd.read_csv(df_patent_loc, dtype={'patent_id': 'string'}, low_memory=False)

    # Read description file with memory optimization
    print("Reading description data...")
    df_desc = pd.read_csv(merged_desc_path, sep='\t', dtype={'patent_id': 'string'})

    # Filter description data to only keep rows with non-empty summary_text
    print("Filtering description data...")
    df_desc_filtered = df_desc[
        (df_desc['summary_text'].notna()) &
        (df_desc['summary_text'] != '') &
        (df_desc['summary_text'].str.strip() != '')
        ][['patent_id', 'summary_text']]

    # Remove duplicates from filtered description data
    print("Removing duplicates from description...")
    df_desc_subset = df_desc_filtered.drop_duplicates(subset=['patent_id'], keep='first')

    # Check if we have any data to merge
    if len(df_patent) == 0 or len(df_desc_subset) == 0:
        print("No data to merge")
        return None

    # Method 1: Use smaller chunks for merging (memory efficient approach)
    print("Performing chunked merge operation...")

    # Create a set of patent_ids from the filtered description data for faster lookup
    desc_patent_ids = set(df_desc_subset['patent_id'].values)

    # Filter patent data to only include those present in description data
    print("Filtering patent data to matching patent_ids...")
    df_patent_filtered = df_patent[df_patent['patent_id'].isin(desc_patent_ids)]

    # Perform merge with smaller datasets
    print("Merging filtered datasets...")
    merged_df = pd.merge(df_patent_filtered, df_desc_subset, on='patent_id', how='left')

    # Handle existing file properly
    output_exists = os.path.exists(output_path)

    if output_exists:
        print("Processing existing file...")
        try:
            # Read existing file with memory constraints
            existing_df = pd.read_csv(output_path, dtype={'patent_id': 'string'})

            # Remove duplicates from existing data based on patent_id
            existing_dedup = existing_df.drop_duplicates(subset=['patent_id'], keep='first')

            # Find new rows that are not in existing file
            if len(merged_df) > 0:
                # Create key sets for comparison
                merged_keys = set(merged_df['patent_id'].values)
                existing_keys = set(existing_dedup['patent_id'].values)

                # Find new rows only
                new_rows = merged_df[~merged_df['patent_id'].isin(existing_keys)]

                if len(new_rows) > 0:
                    print(f"Appending {len(new_rows)} new rows")
                    final_df = pd.concat([existing_dedup, new_rows], ignore_index=True)
                else:
                    final_df = existing_dedup
                    print("No new rows to append")
            else:
                final_df = existing_dedup

        except Exception as e:
            print(f"Error processing existing file: {e}")
            final_df = merged_df
    else:
        # Create new file with merged data
        print("Creating new file")
        final_df = merged_df

    # Save the result
    if len(final_df) > 0:
        final_df.to_csv(output_path, index=False)
        print(f"File saved successfully. Shape: {final_df.shape}")
    else:
        print("No data to save")

    return final_df


# Execute the memory-efficient merge
if __name__ == "__main__":
    try:
        result = memory_efficient_merge()
        if result is not None:
            print(f"Processing completed. Final shape: {result.shape}")
        else:
            print("Processing completed with no data")
    except Exception as e:
        print(f"Error during processing: {e}")
