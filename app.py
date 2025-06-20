import streamlit as st
import pandas as pd
import plotly.express as px
import json
import io
from sentiment_analyzer import analyze_text, get_emoji_for_sentiment
from data_processor import load_data, process_data
from data_visualizer import (
    create_sentiment_distribution_chart,
    create_sentiment_by_platform_chart,
    create_sentiment_wordcloud,
    create_sentiment_over_time_chart
)

# Set page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="./generated-icon.png",
    layout="wide"
)

# App title and description 
st.title("📈 Sentiment Analysis on Social Media")
st.write("""
Analyze sentiment from Instagram, Facebook, and X (Twitter) content.
Upload your data or use the text input to analyze individual posts.
""")

# Sidebar
st.sidebar.title("Options")

# Analysis Options
analysis_option = st.sidebar.radio(
    "Choose Analysis Method:",
    ["Upload Social Media Data", "Analyze Individual Post"]
)

# Initialize session state for storing data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = False

# Function to reset filters
def reset_filters():
    st.session_state.filtered_data = st.session_state.data
    st.session_state.filter_applied = False
    
# Option 1: Upload Social Media Data
if analysis_option == "Upload Social Media Data":
    st.subheader("Upload Social Media Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
    
    # Example datasets option
    example_data = st.checkbox("Use example datasets")
    
    if example_data:
        platform = st.selectbox(
            "Select platform",
            ["Facebook", "Twitter", "Instagram"]
        )
        
        example_file_map = {
            "Facebook": "sample_data/facebook_sample.csv",
            "Twitter": "sample_data/twitter_sample.csv",
            "Instagram": "sample_data/instagram_sample.csv"
        }
        
        try:
            st.session_state.data = load_data(example_file_map[platform])
            st.session_state.filtered_data = st.session_state.data
            st.success(f"Loaded example {platform} dataset")
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
    
    elif uploaded_file is not None:
        try:
            st.session_state.data = load_data(uploaded_file)
            st.session_state.filtered_data = st.session_state.data
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display data and visualizations if data is loaded
    if st.session_state.data is not None:
        # Filtering options
        st.subheader("Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            platforms = st.session_state.data['platform'].unique().tolist()
            selected_platforms = st.multiselect("Platform", platforms, default=platforms)
        
        with col2:
            sentiments = ['positive', 'neutral', 'negative']
            selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
        
        with col3:
            filter_button = st.button("Apply Filters")
            reset_button = st.button("Reset Filters")
        
        if filter_button:
            st.session_state.filtered_data = st.session_state.data[
                (st.session_state.data['platform'].isin(selected_platforms)) &
                (st.session_state.data['sentiment'].isin(selected_sentiments))
            ]
            st.session_state.filter_applied = True
        
        if reset_button:
            reset_filters()
        
        # Display filtered data
        if st.session_state.filtered_data is not None:
            if st.session_state.filter_applied:
                st.write(f"Showing filtered data: {len(st.session_state.filtered_data)} records")
            else:
                st.write(f"Showing all data: {len(st.session_state.filtered_data)} records")
            
            with st.expander("Show Data Table"):
                st.dataframe(st.session_state.filtered_data)
            
            # Display visualizations
            st.subheader("Sentiment Analysis Results")
            
            # Key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                sentiment_counts = st.session_state.filtered_data['sentiment'].value_counts(normalize=True) * 100
                positive_pct = sentiment_counts.get('positive', 0)
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with metric_col2:
                neutral_pct = sentiment_counts.get('neutral', 0)
                st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
            
            with metric_col3:
                negative_pct = sentiment_counts.get('negative', 0)
                st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
            
            # Visualizations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "By Platform", "Over Time", "Word Cloud"])
            
            with tab1:
                dist_chart = create_sentiment_distribution_chart(st.session_state.filtered_data)
                st.plotly_chart(dist_chart, use_container_width=True)
            
            with tab2:
                platform_chart = create_sentiment_by_platform_chart(st.session_state.filtered_data)
                st.plotly_chart(platform_chart, use_container_width=True)
            
            with tab3:
                if 'date' in st.session_state.filtered_data.columns:
                    time_chart = create_sentiment_over_time_chart(st.session_state.filtered_data)
                    st.plotly_chart(time_chart, use_container_width=True)
                else:
                    st.info("Time-based analysis not available for this dataset. Date information is missing.")
            
            with tab4:
                wordcloud = create_sentiment_wordcloud(st.session_state.filtered_data)
                if wordcloud:
                    st.image(wordcloud)
                else:
                    st.info("Word cloud generation requires more text data.")

# Option 2: Analyze Individual Post
else:
    st.subheader("Analyze Individual Social Media Post")
    
    # Platform selection
    platform = st.selectbox(
        "Select platform",
        ["Facebook", "Twitter", "Instagram"]
    )
    
    # Text input
    text_input = st.text_area("Enter post text to analyze:", height=150)
    
    # Analyze button
    if st.button("Analyze"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Perform sentiment analysis
            sentiment, score, compound_score = analyze_text(text_input)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                emoji = get_emoji_for_sentiment(sentiment)
                st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>{sentiment.title()}</h3>", unsafe_allow_html=True)
                
                # Create a gauge chart for sentiment score
                score_gauge = px.pie(values=[compound_score + 1, 2 - (compound_score + 1)], 
                                     names=["Score", ""],
                                     hole=0.7,
                                     color_discrete_sequence=["#1E88E5", "#F0F2F6"])
                score_gauge.update_layout(
                    showlegend=False,
                    annotations=[dict(text=f"{compound_score:.2f}", font_size=20, showarrow=False)]
                )
                st.plotly_chart(score_gauge)
                
            with col2:
                st.markdown("### Sentiment Details")
                st.markdown(f"**Platform:** {platform}")
                st.markdown(f"**Text:** {text_input}")
                st.markdown(f"**Sentiment Score Breakdown:**")
                
                # Create a horizontal bar chart for sentiment components
                components = {
                    "Positive": max(0, score[0]),
                    "Neutral": max(0, score[1]),
                    "Negative": max(0, score[2])
                }
                
                component_df = pd.DataFrame({
                    "Sentiment": list(components.keys()),
                    "Score": list(components.values())
                })
                
                component_chart = px.bar(
                    component_df,
                    x="Score",
                    y="Sentiment",
                    orientation="h",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#4CAF50",
                        "Neutral": "#FFC107",
                        "Negative": "#F44336"
                    }
                )
                component_chart.update_layout(showlegend=False)
                st.plotly_chart(component_chart)
                
                # Interpretation
                st.markdown("### Interpretation")
                if sentiment == "positive":
                    st.markdown("This post expresses a positive sentiment, suggesting approval, happiness, or satisfaction.")
                elif sentiment == "neutral":
                    st.markdown("This post is relatively neutral, stating facts or information without strong emotional content.")
                else:
                    st.markdown("This post expresses a negative sentiment, suggesting disapproval, criticism, or dissatisfaction.")

# Footer
st.markdown("---")
st.markdown("Social Media Sentiment Analysis Tool | Made with Streamlit")
