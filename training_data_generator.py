import random
from shared import *

writeData = True
# Toggle to False to run the code without changing local files.

def allp(iterable, p):
    for i in iterable:
        if not p(i): return False
    return True

def isLower(c):
    return ord(c) >= ord('a') and ord(c) <= ord('z')

words = []
filtered_words = []

with open("words.txt", "r") as f:
    words = f.readlines()

filtered_words = [word for word in words if len(word.strip()) >= MIN_WORD_LENGTH and len(word.strip()) <= MAX_WORD_LENGTH and allp(word.strip(), lambda c: isLower(c))]

random.shuffle(filtered_words)
nTrain = 3*len(filtered_words)//4
train_words = filtered_words[0:nTrain]
test_words = filtered_words[nTrain:]

if writeData:
    with open("train_words.txt", 'w') as f:
        f.writelines(train_words)

    with open("test_words.txt", 'w') as f:
        f.writelines(test_words)


# Now generate gibberish
unwords = []

# Would like to generate words with same length frequency as our real set.
# Count how many we need of each length, then generate until we have enough of each length.
length_counts = { }
for i in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH+1):
    length_counts[i] = sum(1 for w in filtered_words if len(w.strip()) == i)


def randChar():
    return chr(random.randrange(0, 26) + ord('a'))

k = 0
cumulative_english_frequencies = [ k := k + f for f in english_frequencies ]
cumulative_english_frequencies[-1] = 1 # floating point errors makes this not quite 1. do this to prevent index error below.
def randChar_weighted():
    target = random.random()
    i = 0
    while cumulative_english_frequencies[i] < target:
        i = i + 1
    return chr(i + ord('a'))


def randWord(length, weighted = True):
    randFunc = randChar_weighted if weighted else randChar
    return ''.join([randFunc() for _ in range(length)])

for (length, count) in length_counts.items():
    this_length_words = set()
    this_length_real_words = set(w for w in filtered_words if len(w.strip()) == length)
    while len(this_length_words) < count:
        w = randWord(length) + '\n'
        if w in this_length_real_words or w in this_length_words: continue
        this_length_words.add(w)
    unwords.extend(this_length_words)

train_unwords = unwords[0:nTrain]
test_unwords = unwords[nTrain:]

if writeData:
    with open("train_unwords.txt", 'w') as f:
        f.writelines(train_unwords)

    with open("test_unwords.txt", 'w') as f:
        f.writelines(test_unwords)