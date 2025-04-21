import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import re

nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

def clean_word(word):
    """Remove special characters and convert to lowercase."""
   
    word = word.lower()
    word = word.strip()

    return re.sub(r'[^a-zA-Z]','',word)

def analyze_corpus(corpus_words, corpus_sents, name):
    """Analyze a corpus and print statistics."""

    cleaned_words = []

    for word in corpus_words:
        cleaned = clean_word(word)
        if cleaned:  # To skip empty strings
            cleaned_words.append(cleaned)
  
   
    tokens = len(corpus_words)
    types = len(set(corpus_words))
    words_count = len(cleaned_words)
    
    avg_words_per_sent = tokens / len(corpus_sents)
    

    total_length = 0

    for word in cleaned_words:
        word_length = len(word)
        total_length += word_length

    # Avoid zero division
    if words_count > 0:
        avg_word_length = total_length / words_count
    else:
        avg_word_length = 0

    
    
    lemmas = set(lemmatizer.lemmatize(word) for word in cleaned_words)
    lemma_count = len(lemmas)


    freq_dist = FreqDist(cleaned_words)

    pos_tags = [tag for word, tag in nltk.pos_tag(corpus_words)]

    pos_freq = Counter(pos_tags)
    
    top_pos = pos_freq.most_common(10)
    

    print(f"Number of tokens : {tokens}")
    print(f"Numbe r of types : {types}")
    print(f"Number of words (cleaned) : {words_count}")
    print(f" Average words per sentence : {avg_words_per_sent:.2f}")
    print(f" Average word length : {avg_word_length:.2f}")
    print(f" Number of lemmas : {lemma_count}")
    
    print(" Top 10 POS tags:")
    for tag, count in top_pos:
        print(f"{tag}: {count}")
    
    
    return freq_dist


all_words = brown.words()
all_sents = brown.sents()


genre1,genre2 = 'news','fiction'

genre1_words = brown.words(categories=[genre1])
genre1_sents = brown.sents(categories=[genre1])

genre2_words = brown.words(categories=[genre2])
genre2_sents = brown.sents(categories=[genre2])


full_freq_dist = analyze_corpus(all_words, all_sents, "Full Corpus")

genre1_freq_dist = analyze_corpus(genre1_words, genre1_sents, f"{genre1.title()} Genre")
genre2_freq_dist = analyze_corpus(genre2_words, genre2_sents, f"{genre2.title()} Genre")


def plot_freq_distribution(freq_dists, names, log_scale=False):
    
    for i in range(len(freq_dists)):
        freq_dist = freq_dists[i]
        name = names[i]
        

        most_common = freq_dist.most_common()
        freqs = []
        for word, freq in most_common:
            freqs.append(freq)
        
        ranks = list(range(1, len(freqs) + 1)) #Rank the words by their usage
        
        plt.plot(ranks, freqs, label=name)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Word Frequency Distribution (Log-Log Scale)')
        plt.savefig('output/frequency_loglog.png')
    else:
        plt.title('Word Frequency Distribution (Linear Scale)')
        plt.savefig('output/frequency_linear.png')
    
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_freq_distribution([full_freq_dist, genre1_freq_dist, genre2_freq_dist], ["Full Corpus", f"{genre1.title()}", f"{genre2.title()}"], log_scale=False)
plot_freq_distribution([full_freq_dist, genre1_freq_dist, genre2_freq_dist], ["Full Corpus", f"{genre1.title()}", f"{genre2.title()}"], log_scale=True)
