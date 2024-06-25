import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')

def process_text(text):
    # Tokenization and stop word removal
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop]

    # Stemming (using lemma)
    stemmed_tokens = [token.lemma_ for token in doc if not token.is_stop]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

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


