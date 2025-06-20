# Social Media Sentiment Analysis

A powerful Streamlit web application for analyzing and visualizing sentiment from social media content across Instagram, Facebook, and Twitter/X platforms.

## Features

### Data Analysis
- **Multi-platform Support**: Process content from Facebook, Twitter/X, and Instagram
- **Bulk Data Analysis**: Upload CSV or JSON files containing social media data
- **Individual Post Analysis**: Quickly analyze sentiment of specific posts
- **Sample Datasets**: Built-in example datasets for each platform

### Sentiment Analysis
- **Hybrid Algorithm**: Combines NLTK's VADER and TextBlob for improved accuracy
- **Sentiment Classification**: Categorizes content as positive, neutral, or negative
- **Detailed Scoring**: Provides component scores showing positive, neutral, and negative aspects

### Visualizations
- **Sentiment Distribution**: Bar charts showing overall sentiment breakdown
- **Platform Comparisons**: Compare sentiment patterns across different platforms
- **Time-based Analysis**: Track sentiment changes over time
- **Word Clouds**: Visual representation of most common terms by sentiment

## Getting Started

### Step-by-Step Instructions to Run the Application

1. **Prerequisites**
   - Make sure Python 3.6+ is installed on your system
   - Install required libraries:
     ```bash
     pip install streamlit nltk pandas plotly textblob wordcloud numpy
     ```
   - NLTK resources will be downloaded automatically on first run

2. **Running the Application**
   - Clone or download this repository to your local machine
   - Navigate to the project directory in your terminal or command prompt
   - Run the following command:
     ```bash
     streamlit run app.py --server.port 5000
     ```
   - The application will start and open in your default web browser
   - If the browser doesn't open automatically, visit: http://localhost:5000

3. **Troubleshooting**
   - If you encounter any NLTK-related errors, manually download required resources:
     ```python
     import nltk
     nltk.download('vader_lexicon')
     nltk.download('punkt')
     nltk.download('stopwords')
     ```
   - If you face port conflicts, change the port number:
     ```bash
     streamlit run app.py --server.port 8501
     ```

### Using the Application

#### Analyzing Social Media Data
1. Select "Upload Social Media Data" in the sidebar
2. Either upload your own CSV/JSON file or use the example datasets
3. Apply filters to focus on specific platforms or sentiment categories
4. Explore the visualizations in the different tabs

#### Analyzing Individual Posts
1. Select "Analyze Individual Post" in the sidebar
2. Choose the platform (Facebook, Twitter, Instagram)
3. Enter the post text in the text area
4. Click "Analyze" to see detailed sentiment breakdown

## Project Structure

- **app.py**: Main Streamlit application
- **sentiment_analyzer.py**: Sentiment analysis implementation
- **data_processor.py**: Data loading and preprocessing
- **data_visualizer.py**: Visualization components
- **sample_data/**: Example datasets for testing

## Technical Details

### Dependencies
- Streamlit for web interface
- NLTK and TextBlob for sentiment analysis
- Pandas for data manipulation
- Plotly for interactive visualizations
- WordCloud for generating word clouds

### Implementation Notes
- The sentiment analysis uses a hybrid approach combining rule-based (VADER) and machine learning approaches (TextBlob)
- Text preprocessing removes URLs, user mentions, and hashtag symbols
- The application automatically identifies text content columns in uploaded data

## Future Enhancements
- Multilingual sentiment analysis
- Advanced filtering options
- Export of processed data
- Comparative analysis between time periods
- Deeper engagement metrics analysis