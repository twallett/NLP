#%%
import argparse
import os
import pandas as pd
# -----------

# os.mkdir("Text Feature")

path = os.getcwd()

with open("sample.txt", mode = "r") as f:
    text = f.readlines()
#%%

# rows sentences
# columns are number of words in the sentence 

sentences = [line.strip() for line in text]

# Calculate the word count for each sentence
word_counts = [len(sentence.split()) for sentence in sentences]

# Create a pandas DataFrame with sentences and their word counts
df = pd.DataFrame({'Sentence': sentences, 'Word Count': word_counts})

# Print the DataFrame (optional)
print(df)

text_feature_directory = path

# Make sure the directory exists, or create it if it doesn't
if not os.path.exists(text_feature_directory):
    os.makedirs(text_feature_directory)

# Define the CSV file path within the specified directory
csv_file_path = os.path.join(text_feature_directory, 'sent.csv')

# Save the DataFrame to a CSV file in the specified directory
df.to_csv(csv_file_path, index=False)


os.system("Pyhton3 quiz1.py")

# %%
