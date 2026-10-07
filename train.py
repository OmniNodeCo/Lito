import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from training_data import TRAINING_DATA


def train_model(output_file="lito_model.pkl"):
    """
    Train a Naive Bayes text classifier on the training data
    and save the complete pipeline to a .pkl file.
    """
    sentences = []
    labels = []

    # Flatten the training data into two parallel lists
    for intent, examples in TRAINING_DATA.items():
        for example in examples:
            sentences.append(example.lower())
            labels.append(intent)

    print(f"Training on {len(sentences)} examples across {len(TRAINING_DATA)} intents...")

    # Build a pipeline: TF-IDF vectorizer → Naive Bayes classifier
    # This keeps both steps bundled in one object
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),    # Look at single words AND pairs of words
            max_features=5000,     # Keep memory low
            lowercase=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    # Train the model (takes milliseconds)
    model.fit(sentences, labels)

    # Save to file
    joblib.dump(model, output_file)
    file_size = os.path.getsize(output_file) / 1024

    print(f"Model saved to '{output_file}' ({file_size:.1f} KB)")
    print("Training complete!")

    # Quick accuracy test on training data
    accuracy = model.score(sentences, labels)
    print(f"Training accuracy: {accuracy * 100:.1f}%")

    return model


if __name__ == "__main__":
    train_model()