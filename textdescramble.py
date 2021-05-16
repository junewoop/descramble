import string
import math
import random
import numpy as np
import sys

alph_string = string.ascii_lowercase + ' '      # 'abcd...z' + ' '
alph_list = list(alph_string)
alph_size = len(alph_list)
bigram_data = {}
monogram_data = {}

# Read text, replace everything that is not alphabet with " "
def read_text(filename):
    with open(filename, 'r') as file:
        text = file.read().replace('\n',' ')
    return ''.join(x if x in string.ascii_letters else ' ' for x in text).lower()

# Learn "True distribution" from trainfile. monogram_data := pi_hat, bigram_data := q_hat
def create_data(trainfile):
    with open(trainfile, 'r') as file:
        for line in file:
            text = list(line.strip())
            for i in range(len(text)-1):
                if text[i] in string.ascii_letters:
                    monogram_key = text[i].lower()
                    bigram_key = text[i].lower()
                else:
                    monogram_key = " "
                    bigram_key = " "
                if text[i+1] in string.ascii_letters:
                    bigram_key += text[i+1].lower()
                else:
                    bigram_key += " "
                if monogram_key in monogram_data:
                    monogram_data[monogram_key] += 1
                else:
                    monogram_data[monogram_key] = 1
                if bigram_key in bigram_data:
                    bigram_data[bigram_key] += 1
                else:
                    bigram_data[bigram_key]  = 1
            if len(text) > 0:
                if text[len(text)-1] in monogram_data:
                    monogram_data[text[len(text)-1]] += 1
                else:
                    monogram_data[text[len(text)-1]] = 1

# Scramble text using scramble_key.
# If scramble_key = "pq....r", then every 'a' in text is replaced with 'p',
# every 'b' in text is replaced with 'q', and so on.
def scramble(text, scramble_key):
    text = list(text)
    scrambled_text = ""
    for char in text:
        scrambled_text += scramble_key[alph_list.index(char)]
    return scrambled_text

# Count the number of bigrams and monograms occuring in text and return their dict
def count(text):
    monogram_count = {}
    bigram_count = {}
    for i in range(len(text)-1):
        monogram_key = text[i]
        bigram_key = text[i]+text[i+1]
        if monogram_key in monogram_count:
            monogram_count[monogram_key] += 1
        else:
            monogram_count[monogram_key] = 1
        
        if bigram_key in bigram_count:
            bigram_count[bigram_key] += 1
        else:
            bigram_count[bigram_key]  = 1
    if text[len(text)-1] in monogram_count:
        monogram_count[text[len(text)-1]] += 1
    else:
        monogram_count[train_text[len(text)-1]] = 1
    return monogram_count, bigram_count

# Compute the (scaled) bigram energy function of the string obtained by scrambling text with scramble_key
# If a bigram in descrambled string does not appear in train_text, add -log_zero
def bigram_energy_ftn(text, scramble_key, log_zero = 0):
    scrambled_text = scramble(text, scramble_key)
    monogram_count, bigram_count = count(scrambled_text)
    energy = -math.log(monogram_data[scrambled_text[0]])
    for k, v in monogram_count.items():
        energy += v*math.log(monogram_data[k])
    energy -= math.log(monogram_data[scrambled_text[0]])
    energy -= math.log(monogram_data[scrambled_text[len(scrambled_text)-1]])
    for k, v in bigram_count.items():
        if k in bigram_data:
            energy += -v*math.log(bigram_data[k])
        else:
            energy += -v*log_zero
    return energy

# Compute the (scaled) monogram energy function of the string obtained by scrambling text with scramble_key
def monogram_energy_ftn(text, scramble_key):
    scrambled_text = scramble(text, scramble_key)
    monogram_count, _ = count(scrambled_text)
    energy = 0
    for k, v in monogram_count.items():
        if k in monogram_data:
            energy += -v*math.log(monogram_data[k])
    return energy

# Swap random two characters in scramble_key to obtain a new key.
def key_swap(scramble_key):
    i , j = 0, 0
    scramble_key = list(scramble_key)
    while(i == j):
        i = random.randint(0, len(list(scramble_key))-1)
        j = random.randint(0, len(list(scramble_key))-1)
    tmp = scramble_key[i]
    scramble_key[i] = scramble_key[j]
    scramble_key[j] = tmp
    return "".join(scramble_key)

# Do the bigram Metropolis walk on scrambled_text starting with current_key for n times
# The score difference is scaled by beta
# Update best_key and best_energy
def bigram_walk(scrambled_text, current_key, best_key, best_energy, n, beta):
    for i in range(n):
        next_key = key_swap(current_key)
        current_energy = bigram_energy_ftn(scrambled_text, current_key)
        next_energy = bigram_energy_ftn(scrambled_text, next_key)
        walk_probability = math.exp(min(0, -beta*(next_energy-current_energy)))
        if current_energy < best_energy:
            best_key = current_key
            best_energy = current_energy
        if random.random() < walk_probability:
            current_key = next_key
        if i % 500 == 0:
            print(('{:>8}'+scramble(scrambled_text, current_key)[0:99]).format(str(i)+':'))
    return best_key, best_energy

# Do the monogram Metropolis walk on scrambled_text starting with current_key for n times
# The score difference is scaled by beta
# Update best_key and best_energy
def monogram_walk(scrambled_text, current_key, best_key, best_energy, n, beta):
    for i in range(n):
        next_key = key_swap(current_key)
        current_energy = monogram_energy_ftn(scrambled_text, current_key)
        next_energy = monogram_energy_ftn(scrambled_text, next_key)
        walk_probability = math.exp(min(0, -beta*(next_energy-current_energy)))
        if current_energy < best_energy:
            best_key = current_key
            best_energy = current_energy
        if random.random() < walk_probability:
            current_key = next_key
        if i % 500 == 0:
            print(('{:>8}'+scramble(scrambled_text, current_key)[0:99]).format(str(i)+':'))
    return best_key, best_energy
        
       
        

def main():
    # Checking if arguments are valid
    if len(sys.argv) != 6:
        print("Usage: ", sys.argv[0]," [trainfile] [input] [output] [bigram/monogram] [num_iter]")
        return
    else:
        if sys.argv[4] not in ['bigram', 'monogram']:
            print("Usage: ", sys.argv[0]," [trainfile] [input] [output] [bigram/monogram] [num_iter]")
            return
    print("log:", sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

    # Learn "true distribution" from trainfile, read scrambled text
    # Comment out p = ... if you want to start from a random key
    create_data(sys.argv[1])
    scrambled_text = read_text(sys.argv[2])
    p = range(alph_size)
    # p = np.random.permutation(alph_size)
    initial_key = "".join([alph_list[p[z]] for z in range(alph_size)])
    best_key = initial_key

    # Apply Metropolis walk, either bigram or monogram.
    if sys.argv[4] == 'bigram':
        best_energy = bigram_energy_ftn(scrambled_text, best_key)
        best_key, best_energy = bigram_walk(scrambled_text, initial_key, best_key,best_energy, int(sys.argv[5]), 1)
    else:
        best_energy = monogram_energy_ftn(scrambled_text, best_key)
        best_key, best_energy = monogram_walk(scrambled_text, initial_key, best_key, best_energy, int(sys.argv[5]), 1)
    print('key obtained:', '"'+best_key+'"')

    # Descramble and save
    descrambled_text = scramble(scrambled_text, best_key)
    with open(sys.argv[3], 'w') as fp:
        fp.write(descrambled_text)
        fp.close()

    
if __name__ == '__main__':
	main()
        
