import pandas as pd
import json

df = pd.read_csv("hf://datasets/Anthropic/persuasion/persuasion_data.csv")
df.loc[df['source'] != 'Human', 'source'] = 'Claude'
df['rating_initial'] = df['rating_initial'].str[0].astype(int)
df['rating_final'] = df['rating_final'].str[0].astype(int)
df['text'] = df['claim'].astype(str) + " " + df['argument'].astype(str)
df = df.rename(columns={'persuasiveness_metric': 'score', 'worker_id': 'id'})
df = df[['id', 'source', 'text', 'score', 'rating_initial', 'rating_final']]
threshold = 2
# count_above_threshold = (df['score'] > threshold).sum()
# print(f'NUM ROWS: {df.shape[0]}')
# print(f'Above threshold {threshold}: {count_above_threshold}')
# # print(f'Num switch support: {(df['rating_initial'] <= 4 and df['rating_final'] > 4).sum()}')
# print(f'Num switch support: {len(df[(df['rating_initial'] <= 4) & (df['rating_final'] > 4) & (df['persuasiveness_metric'] > threshold)])}')
successful_df = df[((df['rating_initial'] <= 4) & (df['rating_final'] > 4)) | (df['score'] >= threshold)].copy()
unsuccessful_df = df.loc[~df.index.isin(successful_df.index)].copy()
successful_df['is_successful'] = 1
unsuccessful_df['is_successful'] = 0

with open('datasets/processed/AP/successful.json', 'w') as outfile:
    json.dump(successful_df.to_dict(orient='records'), outfile, indent = 4)

with open('datasets/processed/AP/unsuccessful.json', 'w') as outfile:
    json.dump(unsuccessful_df.to_dict(orient='records'), outfile, indent = 4)

final = pd.concat([successful_df, unsuccessful_df], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)
with open('datasets/processed/AP/final.json', 'w') as outfile:
    json.dump(final.to_dict(orient='records'), outfile, indent = 4)

print('Processed AP dataset.')