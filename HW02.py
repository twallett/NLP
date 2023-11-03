#%%

# Write a python script that reads a string from the user input and print the following:

# i. Number of uppercase letters in the string.

astring = "Hello, how R U?"
anumstring = "Today is the 13th of September of 2023!"

i = len([x for x in astring if x == x.upper() and x.isalpha() == True])

# ii. Number of lowercase letters in the string.

ii = len([x for x in astring if x == x.lower() and x.isalpha() == True])

# iii. Number of digits in the string.

iii = len([x for x in anumstring if x.isdigit() == True])

# iv. Number of whitespace characters in the string.

iv = len([x for x in anumstring if x == " "])

# Write a python script that accepts a string then create a new string by shifting one position to left.
# Example: input : class 2021 output: lass 2021c

v = astring[1:] + astring[0]

# Write a python script that a user input his name and program display its initials.

myname = "Tyler Joseph Wallett"

vi = "".join([x for x in myname if x == x.upper() and x.isalpha() == True])

# Write a python script that accepts a string to setup a passwords. The password must have the following requirements

# The password must be at least eight characters long.

# It must contain at least one uppercase letter

# It must contain at least one lowercase letter

# It must contain at least one numeric digit.

CorrectPassword = "Gwutwallett2023"

IncorrectPassword = "gwutwallett"

def PasswordCheck(astr):
    while True:
        
        if len(astr) >= 8:
            pass
        else: 
            print("Not long enough, try again!")
            break
        
        if len([x for x in astr if x == x.upper() and x.isalpha() == True]) >= 1:
            pass
        else: 
            print("No capital letters, try again!")
            break
        
        if len([x for x in astr if x == x.lower() and x.isalpha() == True]) >= 1:
            pass 
        else: 
            print("No lowercase letters, try again!")
            break
        
        if len([x for x in astr if x.isdigit() == True]) >= 1:
            pass 
        else: 
            print("No digits, try again!")
            break
        
        return print("Good password!")
        
PasswordCheck(CorrectPassword)

PasswordCheck(IncorrectPassword)

# Write a python script that reads a given string character by character and count the repeated characters then store it by length of those character(s).

from collections import Counter
        
RepeatedStr = "eeeaao"

Counter(RepeatedStr)

# Write a python script to find all lower and upper case combinations of a given string.
# Example: input: abc output: ’abc’, ’abC’, ’aBc’, ...

from itertools import combinations_with_replacement

astr2 = 'abc'

z = [x for x in astr2] + [x for x in astr2.upper()]

comb = combinations_with_replacement(z,3)

for i in comb:
    print(i)

#%%

# Write a python script that:
# i. Read first n lines of a file.

with open("input.txt", mode='r') as f:
    text = f.readlines()

text = " ".join(text)

# ii. Find the longest words.

adict = {word:len(word) for word in text.split()}

sorted_dict = dict(sorted(adict.items(), key=lambda item: item[1], reverse=True))

for key, value in sorted_dict.items():
    print(f"{key}: {value}")
    
# iii. Count the number of lines in a text file.

with open("input.txt", mode='r') as f:
    line_count = sum(1 for line in f)


# iv. Count the frequency of words in a file.

Counter(text.split()).most_common()

# %%

# =================================================================
# Class_Ex1:
# Write a function that prints all the chars from string1 that appears in string2.
# Note: Just use the strings functionality no other packages should be used.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

str1 = "abc"
str2 = "bcd"

repeated_ch = [x for x in str1 if x in str2]

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Write a function that counts the numbers of a particular letter in a string.
# For example count the number of letter "a" in abstract.
# Note: Compare your function with a count method.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

str3 = "abstract"

cnt = Counter([x for x in str3])
cnt = cnt.get('a')

print(20 * '-' + 'End Q2' + 20 * '-')

# =================================================================
# Class_Ex3:
# Write a function that reads the Story text and finds the strings in the curly brackets.
# Note: You are allowed to use the strings methods
# Copy a text from wiki and add some curly bracket in the text call the string Story.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

story = "Reinforcement learning (RL) is an area of machine learning concerned with \"how intelligent\" agents ought to take \"actions\" in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning."

quotations = " ".join([i for i in story.split() if i.startswith("\"") or i.endswith("\"") ])

print(20 * '-' + 'End Q3' + 20 * '-')

# =================================================================
# Class_Ex4:
# Write a function that read the first n lines of a file.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

with open("input.txt", mode='r') as f:
    text = f.readlines()[:10]

print(20 * '-' + 'End Q4' + 20 * '-')

# =================================================================
# Class_Ex5:
# Write a function that read a file line by line and store it into a list.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

alist = []
with open("input.txt", mode='r') as f:
    text = f.readlines()[:10]
alist.append(text)
alist = alist[0]

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Write a function that read two text files and combine each line from first
# text file with the corresponding line in second text file.
# Use T1.txt and T2.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

text = []
with open("input.txt", mode='r') as f, open("input.txt", mode='r') as g:
    for i, j in zip(f, g):
        text.append("".join([i,j]))
        

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Write a function that creates a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
# Sample output
# function(3)
# ABC
# DEF
# ...
# ...
# ...
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

alphabet = "ABCDEFGHIJKMNLOPQRSTUVWXYZ"

samples = set([alphabet[i:i+2] for i in range(len(alphabet))][::2])

print(20 * '-' + 'End Q7' + 20 * '-')

# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')


with open("input.txt", mode='r') as f:
    text = f.read().split()

length = len(text)

print(20 * '-' + 'End Q8' + 20 * '-')

# =================================================================
# Class_Ex9:
# Write a script that goes over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

astr4 = "oooggf"

sample = [x for x in astr4]

print(20 * '-' + 'End Q9' + 20 * '-')

#%%
# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the list over that index.
# Note: use deque
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

from collections import deque

alist2 = [1,2,3,4,5]

new = deque(alist2)

new.rotate(-2)

print(new)

print(20 * '-' + 'End Q10' + 20 * '-')

#%%
# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

# import os 

# # os.chdir('/Users/tylerwallett/Desktop')

# # print(os.listdir())

print(20 * '-' + 'End Q11' + 20 * '-')

# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

import os

new_file_name = "test.py"
code = """ #%%
print("Hello world!")
"""

with open(new_file_name, 'w') as f:
    f.write(code)
f.close()

print(20 * '-' + 'End Q12' + 20 * '-')

# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

import os 

cwd = os.getcwd()

file_name = "/HWO2.py"

cwd_ = cwd + file_name

asplit = cwd_.split(sep="/")

file_name_ = str([x for x in asplit if ".py" in x][0])

asplit_ = asplit[:-1]

wd = "/".join(asplit_)

print(20 * '-' + 'End Q13' + 20 * '-')

#%%

# Finding dates from corpus 

story = "As of the 13th of September of 1999, I have found tha the crime commited at 12 pm at noon was not true. Therefore, I have changed the date to 5th of August of 1990."

import re

pattern = r"[0-9]+[a-z]+ of [A-Za-z]* of [0-9]*"

for _ in range(2):
    
    astring = re.search(pattern=pattern, string=story)

    idx1, idx2 = astring.span()
    
    print(story[idx1:idx2])
    
    story = story[idx2:]
# %%

story = "As of the Nov. 18, 1989, I have found tha the crime commited at 12 pm at noon was not true. Therefore, I have changed the date to Nov. 18, 1989."

import re

pattern = r"[A-Z][a-z]*. [0-9]+, [0-9]*"

for _ in range(2):
    
    astring = re.search(pattern=pattern, string=story)

    idx1, idx2 = astring.span()
    
    print(story[idx1:idx2])
    
    story = story[idx2:]
# %%
