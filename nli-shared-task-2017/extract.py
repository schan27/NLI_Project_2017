from difflib import SequenceMatcher
from string import punctuation


# define types of tags
INSERT = 'insert'
EQUAL = 'equal'
DELETE_N = 'delete_normal'
DELETE_I = 'delete_inversion'
REPLACE = 'replace'


with open('function-words.txt') as f:
    function_wds = {l.strip() for l in f.readlines() if l}


def tokenize(doc):
    return doc.lower().split()


def peek(stack):
    return stack[-1]


def char_type(c):
    if c in {'a', 'e', 'i', 'o', 'u'}:
        return 'VOWEL'
    return 'CONSONANT'


PATTERNS = {
    (0, 0, 0, 1): INSERT,
    (0, 1, 1, 2): EQUAL,
    (0, 1, 0, 0): DELETE_N,
    (0, 1, 1, 1): DELETE_I,
}


def is_match(code):
    # (tag, i1, i2, j1, j2)
    tag = code[0]
    pattern = code[1:]
    i = pattern[0]
    
    if tag == REPLACE and pattern[:2] == pattern[2:]:
        # 'replace' doesn't require a fixed pattern
        return tag
    
    pattern = tuple(num - i for num in pattern)
    if pattern in PATTERNS and tag in PATTERNS[pattern]:
        return PATTERNS[pattern]
    return False


def get_errors(matcher):
    # there's noise in the errors, so try to be as specific as possible
    codes = matcher.get_opcodes()
    errors = []
    stack = []
    
    corr = matcher.a
    orig = matcher.b
    
    while codes:
        code = codes.pop(0)
        op = code[0]
        i = code[1]  # start index
        
        if not stack:  # empty stack
            if is_match(code) == INSERT:  # can be insertion of inversion
                stack.append((op, i))
            elif is_match(code) == DELETE_N:  # found a deletion
                if corr[i-1] == corr[i]:
                    # repeated letters in corrected string
                    errors.append('ERR_UNDOUBLED_%s' % corr[i])
                else:
                    errors.append('ERR_DELETION_%s' % corr[i])
            elif is_match(code) == REPLACE:  # found a replacement
                errors.append('ERR_REPLACEMENT_%s_%s' % (char_type(corr[i]), orig[i]))
            # do nothing for 'equal'
        else:
            top_op, top_i = peek(stack)
            if top_op == INSERT:
                if is_match(code) == EQUAL and (i == top_i) and codes:
                    stack.append((op, i))  # possible inversion
                else:  # found an insertion
                    if orig[top_i] == orig[top_i+1]:
                        # repeated letters in original string
                        errors.append('ERR_DOUBLED_%s' % orig[top_i])
                    else:
                        errors.append('ERR_INSERTION_%s' % orig[top_i])
                    stack.pop()

            elif top_op == EQUAL:
                try:
                    first_op, first_i = stack.pop(-2)
                    second_op, second_i = stack.pop()
                    if first_op == INSERT and second_op == EQUAL and is_match(code) == DELETE_I:
                        errors.append('ERR_INVERSION')
                except IndexError:
                    pass
                
    return errors


def _extract_errors(original, corrected):
    orig_toks = tokenize(original)
    corr_toks = tokenize(corrected)
    
    errors = []
    for orig, corr in zip(orig_toks, corr_toks):
        if orig != corr:
            matcher = SequenceMatcher(None, corr, orig)
            errors.extend(get_errors(matcher))
            
    return ' '.join(errors)
            
    
def extract_errors(row):
    return _extract_errors(row['original'], row['corrected'])
    
    
def extract_function_words(row):
    return ' '.join(w for w in row['corrected'].split() if w in function_wds)
    
    
def extract_punctuation(row):
    return ' '.join(w for w in row['corrected'].split() if w in punctuation)
    
    
# http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
def find_ngrams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))
  
  
def extract_punc_bigrams(row):
    bigrams = find_ngrams(row['original'].split(), 2)
    punc_bigrams = []
    for w1, w2 in bigrams:
        if w1 in punctuation or w2 in punctuation:
            punc_bigrams.append('%s_%s' % (w1, w2))
        
    return ' '.join(punc_bigrams)
    
    
def extract_punc_trigrams(row):
    trigrams = find_ngrams(row['original'].split(), 3)
    punc_trigrams = []
    for w1, w2, w3 in trigrams:
        if w2 in punctuation:
            punc_trigrams.append('%s_%s_%s' % (w1, w2, w3)) 
        
    
    return ' '.join(['%s_%s_%s' % (w1, w2, w3) for w1, w2, w3 in trigrams if w2 in punctuation])
    
    