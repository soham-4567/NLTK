import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the stopwords from nltk if not done yet
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize the lemmatizer and stopwords
#lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')


# Function to preprocess the text (tokenization, stopword removal, and lemmatization)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if
             word.isalnum() and word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)




df=pd.read_csv(r"C:\Users\Intern\Desktop\Myques.csv")

# Preprocess the questions in the DataFrame
df['Processed Question'] = df['Questions'].apply(preprocess_text)


# Step 2: Define a function to get the most similar question using cosine similarity
def get_answer(user_question):
    # Preprocess the user's question
    processed_user_question = preprocess_text(user_question)

    # Combine all processed questions from the DataFrame and the preprocessed user input
    questions = df['Processed Question'].tolist()
    questions.append(processed_user_question)

    # Use TfidfVectorizer to convert preprocessed text to vectors
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    # Calculate cosine similarity between the user input and all other questions
    similarity_matrix = cosine_similarity(question_vectors[-1], question_vectors[:-1])

    # Get the index of the most similar question
    similar_question_idx = similarity_matrix.argmax()

    # Return the corresponding answer
    return df['Answers'].iloc[similar_question_idx]


# Step 3: Get user input and return the answer
user_input = input("Enter your question: ")
answer = get_answer(user_input)
print(f"Answer: {answer}")