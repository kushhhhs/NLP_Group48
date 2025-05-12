import nltk
from nltk.corpus import brown
import math
from collections import Counter
import pandas as pd
import re 
def clean_word(word):
    """Remove special characters and convert to lowercase."""
   
    word = word.lower()
    word = word.strip()

    return re.sub(r'[^a-zA-Z]','',word)


def calculate_pmi():

    sentences = brown.sents()

    all_words = []
    for sentence in sentences:
        for word in sentence:
            all_words.append(clean_word(word))

    total_words = len(all_words)
    
    print(f"Total words in corpus: {total_words}")

    word_counts = Counter(all_words)

    valid_words = set()
    for word, count in word_counts.items():
        if count >= 10:
            valid_words.add(word)

    print(f"Words occurring at least 10 times: {len(valid_words)}")

    bigram_counts = Counter()
    for sentence in sentences:
        sentence = [clean_word(word) for word in sentence]

        for i in range(len(sentence) - 1):
            if sentence[i] in valid_words and sentence[i+1] in valid_words:
                bigram_counts[(sentence[i], sentence[i+1])] += 1
    
    print(f"Total valid bigrams found: {len(bigram_counts)}")
    
    # PMI for each bigram
    pmi_values = {}
    for (word1, word2), bigram_count in bigram_counts.items():
        p_word1 = word_counts[word1] / total_words
        p_word2 = word_counts[word2] / total_words
        

        p_bigram = bigram_count / (total_words - len(sentences))  
        

        pmi = math.log2(p_bigram / (p_word1 * p_word2))
        pmi_values[(word1, word2)] = pmi
    

    pmi_data = []
    for (w1, w2), pmi in pmi_values.items():
        pmi_data.append({"word1": w1,"word2": w2,"pmi": pmi,"count": bigram_counts[(w1, w2)]})

    pmi_df = pd.DataFrame(pmi_data)
    
    
    highest_pmi = pmi_df.sort_values(by="pmi", ascending=False).head(20)
    lowest_pmi = pmi_df.sort_values(by="pmi", ascending=True).head(20)
    
    print("Word Pairs with Highest PMI")
    print(highest_pmi)
    
    print("Word Pairs with Lowest PMI")
    print(lowest_pmi)
    
    return highest_pmi, lowest_pmi


 
# pd.set_option('display.max_columns', None)
highest_pmi, lowest_pmi = calculate_pmi()
