#%%

# Strings 

s = 'Foo'

print(s)

s = "foo"

print(s)

string = 'ABCD'

print(string)
print(string[0])
print(string[1])
print(string[2])
print(string[3])

#%%

# Slicing strings

S = '0123456'
print(S[1:3])
print(S[:3])
print(S[2:])
print(S[2:3])
print(S[:])
print(S[::-1])
print('The backslash is used for special char \\ \' \"')
print("This is for a new line \n and the we can \t tab")

#%%

# String functionalities

S1 = 'Asegmentofastringiscalledaslice.'
print(len(S1))
print ("*" * 10)
print('ab' in 'absent')
S2 = S1[10:15]
print(S2)
print(S1.find('i'))
print(S1.find('slice'))
print(S1.lower)
print(S1.upper())
print(S1.replace('slice','****'))
print(S1.startswith('a'))
# S1[0] = 'A'
S3 = 'a' + S1[1:]
print(S3)

#%%

# String summary

S1= '1'
S2 = '1abc'
S3 = 'Acd'
S4 = 'abcd. Abcd.'
S5 = '     abcd'
S6 = ' '
print(S4.capitalize())
print(S4.count('c'))
print(S1.isdigit())
print(S2.isdigit())
print(S4.isalnum())
print(S3.encode('UTF-8'))
print(S3.encode('UTF-16'))
print(S1.center(4))
print(S5.strip())
print(S4.index('c'))
print(S6.isspace())
print(S3.istitle())
print('.'.join(S3))
print(S4.split(sep='.'))

#%%

# Open text file

f = open("sample_text.txt", "w+")
for i in range (20):
    f.write('This is a line {} \n'.format(i+1))
f.close()
f = open("sample_text.txt", 'a+')
for i in range (2):
    f.write('Append a line {} \n'.format(i+1))
f.close()

#%%

# writing and reading files

with open("sample_text1.txt",'w', encoding = 'utf-8') as f:
   f.write("Thi is my first file created. \n")
   f.write("This is the second line file\n\n")
   f.write("Last but not least\n")
   f.close()
   
# %% -------------------More on Read-------------------
f = open("sample_text1.txt",'r', encoding = 'utf-8')
f.read(4)    # read the first 4 data
f.read(4)    # read the next 4 data
f.read()     # read in the rest till end of file
f.read()     # further reading returns empty sting
f.tell()     # get the current file position
f.seek(0)    # bring file cursor to initial position
print(f.read())
f.close()

#%%

# reading 

f = open("sample_text1.txt",'r',encoding = 'utf-8')
for line in f:
   print(line)
f.seek(0)
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
f.seek(0)
print(f.readlines())
f.close()

#%%

# Itertools 1

import itertools
import operator

data = [1, 2, 3, 4, 5]
states = ['Newyork', 'Virginia', 'DC', 'Texas']

[print(each) for each in states]

result = itertools.accumulate(data, operator.mul)

for each in result:
    print(each)
    
print(operator.mul(1,9))
print(operator.pow(2,4))

print(help(operator))

result = itertools.combinations(states, 2)

for each in result:
    print(each)
    
for i in itertools.count(10,3):
    print(i)
    if i > 20:
        break
    
#%%

# Itertools 2

import itertools

for i in itertools.count(1,2):
    print(i)
    if i > 20:
        break

states = ['Newyork', 'Virginia', 'DC', 'Texas']

for index, city in enumerate(itertools.cycle(states)):
    print(city)
    if index==10:
        break

S1 = ['A', 'B', 'C', 'D', 'E']
S2 = ['F', 'G', 'H', 'I']

result = itertools.chain(S1, S2)

for each in result:
    print(each)
    
#%%

# Itertools 3

import itertools

S1 = ['A', 'B', 'C', 'D']
selections = [True, False, True, False]

result = itertools.compress(S1, selections)

for each in result:
    print(each)

S2 = itertools.islice(S1, 2)
for each in S2:
    print(each)

S3 = itertools.permutations(S1)
for each in S3:
    print(each)

#%%

# Itertools 4

import itertools

for i in itertools.repeat("spam", 10):
    print(i)

S1 = ['A', 'B', 'C', 'D', 'E',]
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]

for each in itertools.zip_longest(S1, data, fillvalue=None):
    print(each)

result = itertools.combinations_with_replacement(S1, 3)
for each in result:
    print(each)
    
#%%

# os 

# import os

# os.mkdir('test_dir')

# orig_dir = os.getcwd()
# print(orig_dir)

# os.chdir(orig_dir + "\\\\" + 'test_dir')
# new_dir = os.getcwd()
# print(new_dir)

# os.chdir(orig_dir)

# os.rename('test_dir', 'new_test_dir')

# l1 = os.listdir('new_test_dir')
# print(l1)
# l2 = os.listdir(os.curdir)
# print(l2)
# print(l2.sort())

#%%

# Copy remove

# import shutil
# import os
# import subprocess
# shutil.rmtree('new_test_dir')

# os.mkdir('test')
# shutil.copytree('test', 'test1')

# os.system('python 01_Strings.py')
# subprocess.call('python 01_Strings.py', shell=True)

# path = os.path.join(os.getcwd(), 'test')
# print(path)
# foldername, basename = os.path.split(path)
# print(foldername)
# print(basename)

# shutil.rmtree('test')
# shutil.rmtree('test1')
#%%

# Counter

from collections import Counter

l1 = [1,2,3,4,1,2,6,7,3,8,1]
print(Counter(l1))

l2 = ['a', 'b', 'c', 'd', 'a' , 'c', 'c']
print(Counter(l2))

cnt = Counter(l1)
print(cnt[1])
print(cnt.most_common())

cnt = Counter({1:3,2:4})
deduct = {1:1, 2:2}

cnt.subtract(deduct)
print(cnt)
#%%

# Default dic

from collections import defaultdict

nums = defaultdict(int)
nums['one'] = 1
nums['two'] = 2
print(nums['three'])

count = defaultdict(int)
names = "John Julie Jack Ann Mike John John Jack Jack Jen Smith Jen Jen"
list = names.split(sep=' ')
for names in list:
    count[names] +=1
print(count)
#%%

# Order dic

from collections import OrderedDict
from collections import Counter

od = OrderedDict()
od['c'] = 1
od['b'] = 2
od['a'] = 3
print(od)

for key, value in od.items():
    print(key, value)

list = ["a","c","c","a","b","a","a","b","c"]
cnt = Counter(list)
od = OrderedDict(cnt.most_common())
for key, value in od.items():
    print(key, value)
#%%

# Deque

from collections import deque

list = ["a","b","c"]
deq = deque(list)
print(deq)

deq.append("d")
deq.appendleft("e")
print(deq)

deq.pop()
deq.popleft()
print(deq)
print(deq.clear())

list = ["a","b","c"]
deq = deque(list)
print(deq.count("a"))

#%%

# Optional arg

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--output', action='store_true', help="shows output")
# args = parser.parse_args()
# if args.output:
#     print("This is some output")
    
#%%

# Required arg

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--name', required=True)
# args = parser.parse_args()
# print('Hello {}'.format(args.name))
#%%

# runner

# import os
# os.system('python required_arg.py --name Amir')
# os.system('python optional_arg.py -o')
# os.system('python optional_arg.py --output')
# %%
