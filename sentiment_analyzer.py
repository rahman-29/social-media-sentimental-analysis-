import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
import re

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_text(text):
    """
    Analyze sentiment of a text using both VADER and TextBlob.
    Returns sentiment category, component scores, and compound score.
    """
    if not text or text.strip() == "":
        return "neutral", (0.0, 0.0, 0.0), 0.0
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Get VADER sentiment scores
    vader_scores = sia.polarity_scores(cleaned_text)
    
    # Get TextBlob sentiment
    blob = TextBlob(cleaned_text)
    textblob_polarity = blob.sentiment.polarity
    
    # Combine scores (weighted average favoring VADER)
    compound_score = vader_scores['compound'] * 0.7 + textblob_polarity * 0.3
    
    # Determine sentiment category
    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Component scores (positive, neutral, negative)
    component_scores = (vader_scores['pos'], vader_scores['neu'], vader_scores['neg'])
    
    return sentiment, component_scores, compound_score

def clean_text(text):
    """Clean and preprocess text for sentiment analysis."""
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (for Twitter)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_dataframe(df, text_column):
    """
    Add sentiment analysis results to a dataframe.
    Adds 'sentiment', 'sentiment_score', and 'sentiment_components' columns.
    """
    # Ensure we have the text column
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataframe")
    
    # Apply sentiment analysis to each row
    results = []
    for text in df[text_column]:
        sentiment, components, score = analyze_text(text)
        results.append((sentiment, score, components))
    
    # Add results to dataframe
    df['sentiment'] = [r[0] for r in results]
    df['sentiment_score'] = [r[1] for r in results]
    df['sentiment_components'] = [r[2] for r in results]
    
    return df

def get_emoji_for_sentiment(sentiment):
    """Return an appropriate emoji for a sentiment category."""
    if sentiment == "positive":
        return "😊"
    elif sentiment == "negative":
        return "😞"
    else:
        return "😐"
