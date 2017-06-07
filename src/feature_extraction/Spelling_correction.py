import re
import os
import json
from autocorrect import spell

words_re = re.compile(r"(\b[A-Za-z'-]+\b|[.,!?;:()])")

# load the dictionary
DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(DIR, 'dictionary.json')) as dictfile:
    dictionary = json.load(dictfile)
dictionary_words = set(dictionary.keys())


def correct_word(word):    
    if not word.isalpha() or (word.upper() in dictionary_words):
        return word
        
    return spell(word)
    
    
# targets single file or a directory
# if the target is a directory all .txt files in the directory
# will have spelling corrected
# return True on success
# return False on failure
def correct_spelling(target, tokenized=False):
    if os.path.isfile(target):
        return __correct_spelling(target)
    
    elif os.path.isdir(target):
        for filename in os.listdir(target):
            if filename.endswith('.txt'):
                if not __correct_spelling(os.path.join(target, filename)):
                    return False
        
        return True
    else:
        raise FileNotFoundError
            

# target is a single file
# returns True on success
# returns False on failure
def __correct_spelling(target, tokenized=False):
    try:
        output_buffer = []
        with open(target) as infile:
            for line in infile:                
                if tokenized:
                    words = line.split()
                else:
                    words = words_re.findall(line)
                    
                words = map(correct_word, words)
                output_buffer.append(' '.join(words) + '\n')
        
        with open(target + '.corrected', 'w') as outfile:
            outfile.writelines(output_buffer)
        
        return True
        
    except IOError as e:
        print("ERROR: when reading file: " + target + " error message: " + str(e) )
        return False
    
            