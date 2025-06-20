import pandas as pd
import json
import io
import datetime
import re
from sentiment_analyzer import analyze_dataframe

def load_data(file_source):
    """
    Load data from a file source (path or uploaded file).
    Supports CSV and JSON formats.
    """
    if isinstance(file_source, str):  # File path
        if file_source.endswith('.csv'):
            data = pd.read_csv(file_source)
        elif file_source.endswith('.json'):
            data = pd.read_json(file_source)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")
    else:  # Uploaded file
        file_content = file_source.getvalue()
        file_name = file_source.name
        
        if file_name.endswith('.csv'):
            data = pd.read_csv(io.BytesIO(file_content))
        elif file_name.endswith('.json'):
            data = pd.read_json(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")
    
    # Process the data
    return process_data(data)

def process_data(data):
    """
    Process and validate the input data.
    Ensures required columns exist and adds sentiment analysis.
    """
    # Check if data is valid
    if data is None or data.empty:
        raise ValueError("No data found or empty data provided.")
    
    # Check and fix column names
    data.columns = [col.lower().strip() for col in data.columns]
    
    # Identify text column
    text_column = identify_text_column(data)
    if not text_column:
        raise ValueError("Could not identify a text content column in the data.")
    
    # Identify platform column or add it
    if 'platform' not in data.columns:
        # Try to infer platform from data or filename
        if 'source' in data.columns:
            data['platform'] = data['source']
        else:
            # Default to "unknown" for platform
            data['platform'] = "unknown"
    
    # Standardize platform names
    data['platform'] = data['platform'].apply(standardize_platform_name)
    
    # Handle date column if exists
    date_column = next((col for col in data.columns if 'date' in col or 'time' in col), None)
    if date_column:
        try:
            data['date'] = pd.to_datetime(data[date_column])
        except:
            # If conversion fails, create a date column with today's date
            data['date'] = datetime.datetime.now()
    else:
        # Create a date column with today's date
        data['date'] = datetime.datetime.now()
    
    # Add sentiment analysis
    data = analyze_dataframe(data, text_column)
    
    return data

def identify_text_column(data):
    """Attempt to identify the column containing the main text content."""
    # Common text column names
    text_column_options = ['text', 'content', 'post', 'message', 'tweet', 'caption', 'description']
    
    # First, check for exact matches
    for option in text_column_options:
        if option in data.columns:
            return option
    
    # Next, check for partial matches
    for col in data.columns:
        if any(option in col for option in text_column_options):
            return col
    
    # Last resort: look for the column with string data and most characters on average
    string_cols = data.select_dtypes(include=['object']).columns
    
    if not len(string_cols):
        return None
    
    # Calculate average length of text in each string column
    avg_lengths = {}
    for col in string_cols:
        try:
            avg_lengths[col] = data[col].astype(str).apply(len).mean()
        except:
            avg_lengths[col] = 0
    
    # Return column with longest average text length if it exists
    if avg_lengths:
        return max(avg_lengths, key=avg_lengths.get)
    
    return None

def standardize_platform_name(platform):
    """Standardize platform names to Facebook, Twitter, Instagram."""
    platform = str(platform).lower()
    
    if re.search(r'fb|face', platform):
        return 'Facebook'
    elif re.search(r'tw|x\b', platform):
        return 'Twitter'
    elif re.search(r'insta|ig', platform):
        return 'Instagram'
    else:
        return platform.capitalize()
