import pandas as pd
import os

# Load the main file
main_file_path = os.path.join("Data", "g_patents_with_analysis.csv")
df_main = pd.read_csv(main_file_path)

# Load claims files from both locations
claims_file_path_1 = os.path.join("Data", "g_patents_scanner_printer_merged_text.csv")
claims_file_path_2 = os.path.join("Data", "g_patents_scanner_printer_merged_text_2016.csv")

df_claims_1 = pd.read_csv(claims_file_path_1)
df_claims_2 = pd.read_csv(claims_file_path_2)

# Combine both claims dataframes
df_claims_combined = pd.concat([df_claims_1, df_claims_2], ignore_index=True)

# Remove duplicates based on patent_id
df_claims_combined = df_claims_combined.drop_duplicates(subset=['patent_id'], keep='first')

# Convert all patent_id columns to string to ensure consistency
df_main['patent_id'] = df_main['patent_id'].astype(str)
df_claims_combined['patent_id'] = df_claims_combined['patent_id'].astype(str)

# Select only the columns we need from combined claims
df_claims_subset = df_claims_combined[['patent_id', 'merged_claim_text','summary_text']].drop_duplicates()

# Merge the dataframes
df_final = pd.merge(df_main, df_claims_subset, on='patent_id', how='left')

# Remove unwanted columns
unwanted_columns = ['Unnamed: 0', 'merged_claim_text_x', 'cpc_group', 'dependent', 'summary_text_x']
df_final_cleaned = df_final.drop(columns=[col for col in unwanted_columns if col in df_final.columns])

# Rename the columns to remove the "_x" and "_y" suffixes
if 'merged_claim_text_y' in df_final_cleaned.columns:
    df_final_cleaned = df_final_cleaned.rename(columns={'merged_claim_text_y': 'merged_claim_text'})
if 'summary_text_x' in df_final_cleaned.columns:
    df_final_cleaned = df_final_cleaned.rename(columns={'summary_text_x': 'summary_text'})

# Save the final merged file
output_path = os.path.join("Data", "training_data.csv")
df_final_cleaned.to_csv(output_path, index=False)

print(f"Successfully merged all files. Final DataFrame shape: {df_final_cleaned.shape}")
print(f"Final columns: {df_final_cleaned.columns.tolist()}")
