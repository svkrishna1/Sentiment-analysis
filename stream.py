import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import nltk
import re
import seaborn as sns
import pandas as pd
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return 'Positive', sentiment_score
    elif sentiment_score < 0:
        return 'Negative', sentiment_score
    else:
        return 'Neutral', sentiment_score

# Function to summarize text into a single sentence
def summarize_one_sentence(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    if len(sentences) == 1:
        return str(sentences[0])
    else:
        return ' '.join(str(sentences[:1][0]))

# Function to summarize text into two sentences
def summarize_text(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    if len(sentences) <= 2:
        return ' '.join(str(sentence) for sentence in sentences)
    else:
        return ' '.join(str(sentence) for sentence in sentences[:2])

# Function to extract top adjectives for word cloud
def extract_keywords(text, top_n=5):
    blob = TextBlob(text)
    adjectives = [word.lower() for word, pos in blob.tags if pos.startswith('JJ')]
    word_freq = Counter(adjectives)
    return dict(word_freq.most_common(top_n))

# Function to generate word cloud
def generate_word_cloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(keywords)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to plot sentiment distribution
def plot_sentiment_distribution(sentiment_scores):
    if not sentiment_scores:
        st.write("No sentiment scores to display.")
        return
    df = pd.DataFrame(sentiment_scores, columns=['Score'])
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Score'], bins=10, kde=True)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function for Named Entity Recognition (NER)
def named_entity_recognition(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = [f"{' '.join(c[0] for c in chunk)} ({chunk.label()})" for chunk in chunks if isinstance(chunk, Tree)]
    return entities

# Main function for streamlit app
def main():
    st.title("Sentiment Analysis and Text Summarization")
    st.sidebar.title("Input Options")

    # Input options: text or file upload
    input_choice = st.sidebar.radio("Select Input Method", ('Text Input', 'Upload File'))

    # Input text or file
    if input_choice == 'Text Input':
        user_input = st.text_area("Enter your text here", height=150)
    elif input_choice == 'Upload File':
        uploaded_file = st.file_uploader("Choose a file", type="txt")
        if uploaded_file is not None:
            user_input = uploaded_file.read().decode("utf-8")
    
    # Initialize user input
    if not user_input:
        return

    # Remove stopwords automatically
    user_input_cleaned = remove_stopwords(user_input)

    # Sentiment analysis
    sentiment_label, sentiment_score = analyze_sentiment(user_input_cleaned)
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")

    # Summarize full text in one sentence
    st.subheader("One-Sentence Summary")
    summary_one_sentence = summarize_one_sentence(user_input_cleaned)
    st.write(summary_one_sentence)

    # Summarize into 2 sentences
    st.subheader("Two-Sentence Summary")
    summary_two_sentences = summarize_text(user_input_cleaned)
    st.write(summary_two_sentences)

    # Extract keywords for word cloud
    st.subheader("Keyword Extraction for Word Cloud")
    keywords = extract_keywords(user_input_cleaned)
    if keywords:
        st.write("Keywords:", ", ".join(keywords.keys()))
        generate_word_cloud(keywords)
    else:
        st.write("No significant keywords found.")

    # Named Entity Recognition
    st.subheader("Named Entities")
    entities = named_entity_recognition(user_input)
    if entities:
        st.write("Entities:", ", ".join(entities))
    else:
        st.write("No named entities found.")

    # Show statistics
    st.subheader("Text Statistics")
    original_sentences = len(TextBlob(user_input_cleaned).sentences)
    original_words = len(user_input_cleaned.split())
    summary_sentences = len(summary_two_sentences.split('.'))
    summary_words = len(summary_two_sentences.split())
    st.write(f"Original Text Sentences: {original_sentences}")
    st.write(f"Original Text Words: {original_words}")
    st.write(f"Summary Text Sentences: {summary_sentences}")
    st.write(f"Summary Text Words: {summary_words}")

    # Option to upload multiple reviews and plot sentiment distribution
    st.subheader("Sentiment Distribution Visualization")
    if st.sidebar.checkbox("Upload Multiple Reviews for Sentiment Distribution"):
        uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)
        if uploaded_files:
            sentiment_scores = []
            for file in uploaded_files:
                text = file.read().decode("utf-8")
                cleaned_text = remove_stopwords(text)
                _, score = analyze_sentiment(cleaned_text)
                sentiment_scores.append(score)
            plot_sentiment_distribution(sentiment_scores)

if __name__ == "__main__":
    main()
