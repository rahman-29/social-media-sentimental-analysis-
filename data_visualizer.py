import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from io import BytesIO
import base64
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def create_sentiment_distribution_chart(data):
    """Create a bar chart showing the distribution of sentiments."""
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Map colors to sentiments
    color_map = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Order sentiments
    sentiment_order = ['positive', 'neutral', 'negative']
    sentiment_counts['Sentiment'] = pd.Categorical(
        sentiment_counts['Sentiment'],
        categories=sentiment_order,
        ordered=True
    )
    sentiment_counts = sentiment_counts.sort_values('Sentiment')
    
    # Create the chart
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map=color_map,
        title='Sentiment Distribution',
        labels={'Count': 'Number of Posts', 'Sentiment': 'Sentiment Category'},
        text='Count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='Number of Posts',
        xaxis={'categoryorder': 'array', 'categoryarray': sentiment_order}
    )
    
    return fig

def create_sentiment_by_platform_chart(data):
    """Create a grouped bar chart showing sentiment distribution by platform."""
    # Get sentiment counts by platform
    platform_sentiment = data.groupby(['platform', 'sentiment']).size().reset_index()
    platform_sentiment.columns = ['Platform', 'Sentiment', 'Count']
    
    # Map colors to sentiments
    color_map = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Order sentiments
    sentiment_order = ['positive', 'neutral', 'negative']
    platform_sentiment['Sentiment'] = pd.Categorical(
        platform_sentiment['Sentiment'],
        categories=sentiment_order,
        ordered=True
    )
    
    # Create the chart
    fig = px.bar(
        platform_sentiment,
        x='Platform',
        y='Count',
        color='Sentiment',
        color_discrete_map=color_map,
        title='Sentiment Distribution by Platform',
        labels={'Count': 'Number of Posts', 'Platform': 'Social Media Platform'},
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title='Platform',
        yaxis_title='Number of Posts',
        legend_title='Sentiment'
    )
    
    return fig

def create_sentiment_over_time_chart(data):
    """Create a line chart showing sentiment over time."""
    # Ensure we have a date column
    if 'date' not in data.columns:
        return None
    
    # Group by date and sentiment, and count occurrences
    data['date'] = pd.to_datetime(data['date']).dt.date
    time_sentiment = data.groupby(['date', 'sentiment']).size().reset_index()
    time_sentiment.columns = ['Date', 'Sentiment', 'Count']
    
    # Map colors to sentiments
    color_map = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Create the chart
    fig = px.line(
        time_sentiment,
        x='Date',
        y='Count',
        color='Sentiment',
        color_discrete_map=color_map,
        title='Sentiment Trends Over Time',
        labels={'Count': 'Number of Posts', 'Date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Posts',
        legend_title='Sentiment'
    )
    
    return fig

def create_sentiment_wordcloud(data, max_words=100):
    """
    Create a word cloud from the text data, colored by sentiment.
    Returns an image as base64 encoded string.
    """
    if 'text' not in data.columns and 'content' not in data.columns:
        # Try to find a text column
        text_cols = [col for col in data.columns if col in ['text', 'content', 'message', 'post', 'tweet', 'caption']]
        if not text_cols:
            return None
        text_col = text_cols[0]
    else:
        text_col = 'text' if 'text' in data.columns else 'content'
    
    # Get text data
    all_text = " ".join(data[text_col].astype(str))
    
    # Clean text
    all_text = re.sub(r'http\S+|www\S+|https\S+', '', all_text, flags=re.MULTILINE)
    all_text = re.sub(r'@\w+', '', all_text)
    all_text = re.sub(r'#(\w+)', r'\1', all_text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add common social media terms to stopwords
    social_media_terms = {'rt', 'like', 'follow', 'retweet', 'post', 'facebook', 'twitter', 'instagram', 'comment'}
    stop_words.update(social_media_terms)
    
    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=max_words,
            collocations=False
        ).generate(all_text)
        
        # Convert to image
        img = wordcloud.to_image()
        
        # Save image to BytesIO object
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Encode image to base64 string
        encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{encoded_img}"
    except Exception as e:
        print(f"Error generating word cloud: {str(e)}")
        return None
