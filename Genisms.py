import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    # Tokenization using NLTK
    tokens = word_tokenize(text.lower())  # Use NLTK's word_tokenize for tokenization
    
    # Remove stop words using Gensim
    tokens_no_stopwords = remove_stopwords(' '.join(tokens)).split()
    
    # Stemming using NLTK
    stemmed_tokens = [stemmer.stem(token) for token in tokens_no_stopwords]
    
    # Lemmatization using NLTK
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_no_stopwords]
    
    return tokens, stemmed_tokens, lemmatized_tokens

# Example usage
if __name__ == "__main__":
    # Accept input text from the user
    user_text = input("Enter a sentence or paragraph: ")
    
    # Process the user input
    tokens, stemmed_tokens, lemmatized_tokens = process_text(user_text)
    
    # Print results
    print("\nOriginal Tokens:")
    print(tokens)
    
    print("\nStemmed Tokens:")
    print(stemmed_tokens)
    
    print("\nLemmatized Tokens:")
    print(lemmatized_tokens)



