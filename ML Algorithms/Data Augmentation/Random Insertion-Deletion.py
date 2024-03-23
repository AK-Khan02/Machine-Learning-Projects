import random

def random_insertion(sentence, n):
    words = sentence.split()
    for _ in range(n):
        synonyms = []
        word_to_replace = random.choice(words)
        for syn in wordnet.synsets(word_to_replace):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        synonyms = list(set(synonyms))
        if synonyms:
            synonym = random.choice(synonyms)
            insert_position = random.randint(0, len(words))
            words.insert(insert_position, synonym)
    return ' '.join(words)

def random_deletion(sentence, p=0.1):
    if len(sentence.split()) == 1:  # return if single word
        return sentence
    words = sentence.split()
    remaining = list(filter(lambda x: random.uniform(0, 1) > p, words))
    if len(remaining) == 0:  # if all words are deleted, choose a random word
        return random.choice(words)
    else:
        return ' '.join(remaining)
