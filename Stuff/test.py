#%%

import os
import subprocess

path = os.getcwd()

code = """#%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', action='store_true', help="shows output")
args = parser.parse_args()

if args.output:
    print("This is some output")
"""

with open("new.py", mode="w") as f:
    f.write(code)

# %%
os.system("ls")
# %%

subprocess.call("Python3 new.py -o", shell=True)

#%%


os.remove(path + "/new.py")


# %%
