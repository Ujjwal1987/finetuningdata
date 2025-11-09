import pandas as pd
import os

merged_desc = os.path.join("Data", "g_brf_sum_text_2016.tsv")
df_desc = pd.read_csv(merged_desc, sep='\t')

df_patent_loc = os.path.join("Data", "g_patents_printer_merged_text.csv")
df_patent = pd.read_csv(df_patent_loc)

df2_subset = df_desc[['patent_id', 'summary_text']]

merged_df = pd.merge(df_patent, df_desc, on='patent_id', how='left')

merged_df.to_csv(r'Data/g_patents_printer_merged_text.csv')
