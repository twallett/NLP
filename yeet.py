#%%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()

print(f"hello {args.name}")