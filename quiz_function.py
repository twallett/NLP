#%%
import argparse
import os
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input text file")
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
with open(args.input_file, 'r') as f:
    text = f.read()
# ----------------------------------------------------------------------------------------------------------------------
sentences = []
current_sentence = ''
for char in text:
    if char in '.!?':
        sentences.append(current_sentence)
        current_sentence = ''
    else:
        current_sentence += char
sentences = [s.strip() for s in sentences if s.strip()]
# ----------------------------------------------------------------------------------------------------------------------
df = pd.DataFrame()
df['sentence'] = sentences
df['num_words'] = df['sentence'].apply(lambda x: len(x.split()))
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
folder_name = "Text Feature"
os.makedirs(folder_name, exist_ok = True)
df.to_csv(os.getcwd()+ os.sep + folder_name + os.sep +'word.csv')

# %%
