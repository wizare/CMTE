import nltk
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


def base_tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def simp_zh_tokenize(string):
    return base_tokenize(string, [str.split, remove_nullchar])


def kw_en_tokenize(string):
    return base_tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def simp_en_tokenize(string):
    return base_tokenize(string, [nltk_tokenize, lower])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)


def truecasing(tokens):
    ret = []
    is_start = True
    for word, tag in tokens:
        if word == 'i':
            ret.append('I')
        elif tag[0].isalpha():
            if is_start:
                ret.append(word[0].upper() + word[1:])
            else:
                ret.append(word)
            is_start = False
        else:
            if tag != ',':
                is_start = True
            ret.append(word)
    return ret


def remove_nullchar(tokens):
    return [x for x in tokens if (x != '') and (len(x) > 0)]
