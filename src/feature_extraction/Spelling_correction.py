import re
import os
import json

words_re = re.compile(r"(\b[A-Za-z'-]+\b|[.,!?;:()])")
DIR = os.path.abspath(os.path.dirname(__file__))

# load the dictionary
with open(os.path.join(DIR, 'dictionary.json')) as dictfile:
        dictionary = json.load(dictfile)
dictionary_words = set(dictionary.keys())


def correct_word(word):  
    from autocorrect import spell
    if not word.isalpha() or (word.upper() in dictionary_words):
        return word
        
    return spell(word)
    
    
def correct_spelling(target):
    if os.path.isfile(target):
        return __correct_spelling(target)


# target is a single file
# returns True on success
# returns False on failure
def __correct_spelling(target, tokenized=False):
    output_dir = os.path.join(os.path.dirname(target), 'corrected_out')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
            
        with open(os.path.join(output_dir, os.path.basename(target) + '.corrected'), 'w') as outfile:
            outfile.writelines(output_buffer)
        
        return True
    
    except Exception as e:
        print("ERROR: when reading file: " + target + " error message: " + str(e) )
        return False
