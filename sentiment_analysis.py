import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK data (only first time)
nltk.download('punkt')

# Sample dataset
data = [
    ("I love this product, it's fantastic!", "positive"),
    ("Absolutely wonderful experience.", "positive"),
    ("Worst service ever!", "negative"),
    ("I hate it, very disappointing.", "negative"),
    ("It is okay, nothing special.", "neutral"),
    ("Not bad, but could be better.", "neutral"),
    ("Really enjoyed the service!", "positive"),
    ("Terrible customer support.", "negative"),
    ("Pretty decent, would recommend.", "positive"),
    ("Average experience, not impressed.", "neutral")
]

# Split data
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predict
y_pred = model.predict(X_test_vectors)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Predict custom input
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Try it
while True:
    user_input = input("\nEnter text to analyze sentiment (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")
