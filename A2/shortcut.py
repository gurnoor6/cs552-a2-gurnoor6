import itertools
import jsonlines
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('uh')

import string
puncs = string.punctuation

def word_pair_extraction(prediction_files, tokenizer):
    '''
    Extract all word pairs (word_from_premise, word_from_hypothesis) from input as features.
    
    INPUT: 
      - prediction_files: file path for all predictions
      - tokenizer: tokenizer used for tokenization
    
    OUTPUT: 
      - word_pairs: a dict of all word pairs as keys, and label frequency of values. 
    '''
    word_pairs = {}
    label_to_id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    
    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            for pred in reader.iter():
                #########################################################
                #          TODO: construct word_pairs dictionary        # 
                #  - tokenize the text with 'tokenizer'                 # 
                #  - pair words as keys (you can use itertools)         #
                #  - count predictions for each paired words as values  # 
                #  - remenber to filter undesired word pairs            # 
                #########################################################
                # Replace "..." statement with your code
                p_tokens = tokenizer.tokenize(pred['premise'])
                h_tokens = tokenizer.tokenize(pred['hypothesis'])

                p_tokens = list(filter(lambda x: x not in stop_words and x not in puncs and not x.startswith('##'), p_tokens))
                h_tokens = list(filter(lambda x: x not in stop_words and x not in puncs and not x.startswith('##'), h_tokens))

                pairs = itertools.product(p_tokens, h_tokens)
                pairs = set(filter(lambda x: x[0] != x[1], pairs))
                prediction = pred['prediction']
                tup = (prediction == 'entailment', prediction == 'neutral', prediction == 'contradiction')
                for pair in pairs:
                    word_pairs[pair] = tuple(map(sum, zip(word_pairs.get(pair, (0, 0, 0)), tup)))
                
                #####################################################
                #                   END OF YOUR CODE                #
                #####################################################
    
    return word_pairs
