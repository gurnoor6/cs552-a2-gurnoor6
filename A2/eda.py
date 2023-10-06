import random
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
  
from nltk.corpus import wordnet, stopwords

# ========================== Synonym Replacement ========================== #
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(sentence, n):
    
    words = sentence.split()
    
    ############################################################################
    # TODO: Replace up to n random words in the sentence with their synonyms.  #
    #   You should                                                             #
    #   - (i)   replace random words with one of its synonyms, until           #
    #           the number of replacement gets to n or all the words           #
    #           have been replaced;                                            #
    #   - (ii)  NO stopwords should be replaced!                               #
    #   - (iii) return a new sentence after all the replacement.               #
    ############################################################################
    # Replace "..." with your code
    non_stopwords = list(filter(lambda x: x not in stopwords.words('english'), words))
    random_word_list = random.sample(non_stopwords, min(n, len(non_stopwords)))
    sample_synonyms = lambda x: random.sample(get_synonyms(x), 1)[0] if len(get_synonyms(x)) else x
    word_mapping = {k: sample_synonyms(k) for k in random_word_list}
    new_sentence = " ".join([word_mapping.get(word, word) for word in words])
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(len(words)-1, max_deletion_n)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    ############################################################################
    # TODO: Randomly delete words with probability p. You should               #
    # - (i)   iterate through all the words and determine whether each of them #
    #         should be deleted;                                               #
    # - (ii)  you can delete at most `max_deletion_n` words;                   #
    # - (iii) return the new sentence after deletion.                          #
    ############################################################################
    # Replace "..." with your code
    deletion_indices = [i for i in range(len(words)) if random.uniform(0, 1) <= p]
    deletion_indices = deletion_indices[:max_deletion_n]

    new_sentence = " ".join([word for i, word in enumerate(words) if i not in deletion_indices])
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    
    words = sentence.split()
    assert len(words) > 1, 'The sentence should have more than two words!'
    ############################################################################
    # TODO: Randomly swap two words in the sentence. You should                #
    # - (i)   iterate through all the words and determine whether each of them #
    #         should be deleted;                                               #
    # - (ii)  you can delete at most `max_deletion_n` words;                   #
    # - (iii) return the new sentence after deletion.                          #
    ############################################################################
    # Replace "..." with your code
    random_idx_1, random_idx_2 = random.sample(range(len(words)), 2)
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    new_sentence = " ".join(words)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence

# ========================== Random Insertion ========================== #
def random_insertion(sentence, n):
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence

def add_word(new_words):
    
    synonyms = []
    ############################################################################
    # TODO: Randomly choose one synonym and insert it into the word list.      #
    # - (i)  Get a synonym word of one random word from the word list;         #
    # - (ii) Insert the selected synonym into a random place in the word list. #
    ############################################################################
    # Replace "..." with your code
    synonyms = []
    is_synonyms = [get_synonyms(word) for word in new_words]
    if not any(is_synonyms):
      return
    while not synonyms:
      synonyms = get_synonyms(random.sample(new_words, 1)[0])
    random_synonym = synonyms[0]
    new_words.insert(random.randint(0, len(new_words)), random_synonym)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
