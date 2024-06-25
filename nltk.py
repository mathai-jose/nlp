import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary resources (uncomment if not already downloaded)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

def process_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens, stemmed_tokens, lemmatized_tokens

# Example usage
if name == "main":
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