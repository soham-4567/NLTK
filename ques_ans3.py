import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic tokens and stopwords, then lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens



df=pd.read_csv(r"C:\Users\Intern\Desktop\Myques.csv")
# Preprocess the questions
df['Processed_Question'] = df['Questions'].apply(preprocess_text)

# Prepare data for Word2Vec
sentences = df['Processed_Question'].tolist()

# Train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)


def get_sentence_vector(tokens, model):
    # Initialize an empty vector
    vector = np.zeros(model.vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    if count != 0:
        vector = vector / count
    return vector


# Create sentence vectors for all questions
df['Vector'] = df['Processed_Question'].apply(lambda x: get_sentence_vector(x, model))


def get_answer(user_question):
    # Preprocess the user question
    processed_user_question = preprocess_text(user_question)
    user_vector = get_sentence_vector(processed_user_question, model).reshape(1, -1)

    # Prepare all question vectors
    question_vectors = np.vstack(df['Vector'].values)

    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, question_vectors).flatten()

    # Get the index of the most similar question
    best_idx = similarities.argmax()
    best_similarity = similarities[best_idx]

    # Optionally, set a similarity threshold
    threshold = 0.5  # Adjust as needed
    if best_similarity < threshold:
        return "I'm sorry, I don't understand the question."

    # Return the corresponding answer
    return df['Answers'].iloc[best_idx]


def main():
    print("Welcome to the Q&A System. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = get_answer(user_input)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()