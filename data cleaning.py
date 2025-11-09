import pandas as pd
import numpy as np
import os

# Define the path to the file within the "Data" folder
# Make sure the "Data" folder is in the same directory as your Python script
file_path = os.path.join("Data", "g_claims_2016.tsv")
cpc_file_path = os.path.join("Data", "g_cpc_current.tsv")

try:
    # Read the TSV file using read_csv with the tab separator
    # The pandas library is efficient and can handle large files. [1, 2, 4, 9]
    df = pd.read_csv(file_path, sep='\t')


    # Get the column names from the DataFrame's columns attribute. [6]
    column_names = df.columns.tolist()

    # Print the column names
    print("Column Names:")
    for name in column_names:
        print(name)

        # Filter the DataFrame to remove rows where 'patent_id' starts with "D"
        # The '~' symbol negates the condition, so we keep rows that DO NOT start with "D"
    prefixes_to_remove = ['D', 'PP', 'RE']
    condition_prefixes = ~df['patent_id'].str.startswith(tuple(prefixes_to_remove), na=False)
    condition_blanks = pd.notna(df['patent_id'].replace(r'^\s*$', np.nan, regex=True))
    filtered_df = df[condition_prefixes & condition_blanks]

    # Display the number of rows after filtering
    print(f"\nNumber of rows after removing patent_ids starting with 'D': {len(filtered_df)}")

    df_cpc = pd.read_csv(cpc_file_path, sep='\t')
    if 'cpc_group' not in df_cpc.columns:
        print("Error: 'cpc_group' column not found in the CPC data. Check the file structure.")
    else:
        df_cpc_subset = df_cpc[['patent_id', 'cpc_group']]
        df_merged_with_cpc_group = pd.merge(
            left=filtered_df,
            right=df_cpc_subset,
            on='patent_id',
            how='left'  # Use 'left' to keep all claims even if they lack CPC data
        )

    # Save the filtered DataFrame to a new TSV file
    # We use index=False to avoid writing the DataFrame index as a new column in the file. [2, 5, 6, 9]
    df_merged_with_cpc_group.to_csv(r"Data/g_claims_2015.csv", index=False)

    print(f"\nSuccessfully filtered the data and saved it to: {'Data/g_claims_2015.csv'}")

except KeyError:
    print("Error: A column named 'patent_id' was not found in the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")