import nltk
from nltk.corpus import wordnet
from random import randint

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    """
    Return WORDNET POS compliance to WORDNET lemmatization (a, n, r, v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def synonym_replacement(sentence, n):
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    new_words = words.copy()

    for _ in range(n):
        word_to_replace = randint(0, len(words) - 1)
        synonyms = []

        word, tag = tagged[word_to_replace]
        wordnet_pos = get_wordnet_pos(tag)  # Convert POS tag to format wordnet.synsets() expects

        if wordnet_pos is not None:
            for syn in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())

        synonyms = list(set(synonyms))

        if synonyms:
            synonym = np.random.choice(synonyms)
            new_words[word_to_replace] = synonym

    return ' '.join(new_words)
