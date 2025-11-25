import pandas as pd
import os

merged_desc = os.path.join("Data", "g_brf_sum_text_2016.tsv")
df_desc = pd.read_csv(merged_desc, sep='\t', low_memory=False)

df_patent_loc = os.path.join("Data", "g_patents_merged_text_2016.csv")
df_patent = pd.read_csv(df_patent_loc)

# Debug: Check the column names and data types
print("Patent file columns:", df_patent.columns.tolist())
print("Description file columns:", df_desc.columns.tolist())

# Check if patent_id columns have the same name and data type
print("Patent ID column dtype:", df_patent['patent_id'].dtype if 'patent_id' in df_patent.columns else "Not found")
print("Summary text column dtype:", df_desc['summary_text'].dtype if 'summary_text' in df_desc.columns else "Not found")

# Ensure patent_id columns are consistent (convert to string)
if 'patent_id' in df_patent.columns:
    df_patent['patent_id'] = df_patent['patent_id'].astype(str)
if 'patent_id' in df_desc.columns:
    df_desc['patent_id'] = df_desc['patent_id'].astype(str)

# Check for any NaN values in patent_id
print("NaN in patent_id (patent):", df_patent['patent_id'].isna().sum() if 'patent_id' in df_patent.columns else 0)
print("NaN in patent_id (desc):", df_desc['patent_id'].isna().sum() if 'patent_id' in df_desc.columns else 0)

# Create subset with only the columns we need
df_desc_subset = df_desc[['patent_id', 'summary_text']].drop_duplicates()

# Merge on patent_id
merged_df = pd.merge(df_patent, df_desc_subset, on='patent_id', how='left')

# Check if summary_text is populated
print("Summary text null count:",
      merged_df['summary_text'].isna().sum() if 'summary_text' in merged_df.columns else "Column not found")
print("Sample of merged data:")
print(merged_df[['patent_id', 'summary_text']].head(10))

# Check if there are any empty/NaN summary_text values
if 'summary_text' in merged_df.columns:
    empty_summary_count = merged_df['summary_text'].isna().sum() + (merged_df['summary_text'] == '').sum()
    print(f"Empty/NaN summary_text count: {empty_summary_count}")

    # Check if all summary_text entries are empty
    if len(merged_df) > 0 and empty_summary_count == len(merged_df):
        print("WARNING: All summary_text entries are empty/NaN. Not appending to file.")
        exit()
    elif empty_summary_count > 0:
        print(f"Warning: {empty_summary_count} rows have empty/NaN summary_text")

# Check if output file already exists
output_path = os.path.join("Data", "g_patents_scanner_printer_merged_text.csv")

if os.path.exists(output_path):
    # File exists, read it and append new rows
    existing_df = pd.read_csv(output_path)

    # Remove duplicates from existing data based on patent_id to avoid duplicate keys
    existing_df_dedup = existing_df.drop_duplicates(subset=['patent_id'], keep='first')

    # Find rows that are not already in the existing file
    # Create a key column for comparison (assuming patent_id is the unique identifier)
    merged_df['key'] = merged_df['patent_id']
    existing_df['key'] = existing_df['patent_id']

    # Find new rows (rows in merged_df that are not in existing_df)
    new_rows = merged_df[~merged_df['key'].isin(existing_df['key'])]

    # Check if new rows have non-empty summary_text
    if len(new_rows) > 0:
        # Filter out rows with empty/NaN summary_text
        valid_new_rows = new_rows[
            (new_rows['summary_text'].notna()) &
            (new_rows['summary_text'] != '') &
            (new_rows['summary_text'].str.strip() != '')
            ]

        if len(valid_new_rows) > 0:
            print(f"Appending {len(valid_new_rows)} new rows with valid summary_text")
            # Combine existing data with new rows
            final_df = pd.concat([existing_df, valid_new_rows], ignore_index=True)
            final_df.to_csv(output_path, index=False)
            print("File updated successfully")
        else:
            print("No new rows with valid summary_text to append")
    else:
        print("No new rows to append - file already contains all data")
        final_df = existing_df  # Keep existing data
else:
    # File doesn't exist, create it
    print("Creating new file")
    merged_df.to_csv(output_path, index=False)

print(f"Merged successfully. Shape: {merged_df.shape}")
