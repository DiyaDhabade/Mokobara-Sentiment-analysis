import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
import plotly.express as px
import seaborn as sns
import os
import random

# Set custom font to Inter (if downloaded and placed in fonts/)
try:
    font_dirs = ['fonts/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Inter'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# Mokobara brand colors
mokobara_palette = {
    'Positive': '#F5BF00',   # Mokobara Gold
    'Neutral': '#B0B0B0',    # Soft Grey
    'Negative': '#1D1D1D'    # Charcoal
}

# Extended palette for WordCloud
wordcloud_colors = ['#F5BF00', '#2D2D2D', '#C9B26B', '#7C7C7C', '#B0B0B0']

def mokobara_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(wordcloud_colors)

sns.set_palette([mokobara_palette['Negative'], mokobara_palette['Neutral'], mokobara_palette['Positive']])

# Load the Excel file
FILE_PATH = "mokobara_reviews.csv.xlsx"
if not os.path.exists(FILE_PATH):
    st.error(f"File '{FILE_PATH}' not found. Please make sure it is in the same directory.")
    st.stop()

df = pd.read_excel(FILE_PATH)

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['review_body'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
# df['sentiment_label'] = df['sentiment_scores'].apply(lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral'))
# Assign initial sentiment labels
df['sentiment_label'] = df['sentiment_scores'].apply(lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral'))

# Override: If rating is < 4 and sentiment is positive, mark as Negative
df.loc[(df['rating'] < 4) & (df['sentiment_label'] == 'Positive'), 'sentiment_label'] = 'Negative'

# Streamlit App Configuration
st.set_page_config(page_title="Mokobara Review Dashboard", layout="wide")
st.title("🧳 Mokobara Review Sentiment Dashboard")

# Word Cloud
st.subheader("☁️ Word Cloud of All Reviews")
text = " ".join(df['review_body'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=mokobara_color_func).generate(text)
st.image(wordcloud.to_array(), use_container_width =True)

# Ratings Distribution
st.subheader("⭐ Rating Distribution")
fig_rating = px.histogram(df, x="rating", nbins=5, title="Distribution of Ratings",
                          color_discrete_sequence=['#FFC211'])
fig_rating.update_layout(font=dict(family="Inter", size=14))
st.plotly_chart(fig_rating, use_container_width=True)

# Sentiment Distribution by Product
st.subheader("📊 Sentiment Distribution by Product")
sentiment_counts = df.groupby(['product_name', 'sentiment_label']).size().reset_index(name='count')
fig_sentiment = px.bar(sentiment_counts, x='product_name', y='count', color='sentiment_label',
                       color_discrete_map=mokobara_palette,
                       barmode='group', title="Sentiment Counts per Product")
fig_sentiment.update_layout(font=dict(family="Inter", size=14), xaxis_tickangle=-45)
st.plotly_chart(fig_sentiment, use_container_width=True)

# Scatter Plot: Sentiment Score vs Rating
st.subheader("📉 Sentiment Score vs Star Rating")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='rating', y='sentiment_scores', data=df, hue='sentiment_label', palette=mokobara_palette, ax=ax1)
ax1.set_title('Sentiment Score vs Star Rating')
ax1.set_xlabel('Star Rating')
ax1.set_ylabel('Sentiment Score')
ax1.axhline(0, color='gray', linestyle='--')
st.pyplot(fig1)

# Boxplot: Star Rating Distribution by Product
st.subheader("📦 Rating Distribution by Product")
fig2, ax2 = plt.subplots(figsize=(10, 6))
# Define a custom color palette
boxplot_palette = ['#F5BF00', '#D9C082', '#B0B0B0', '#7C7C7C', '#2D2D2D']
# Use it in the boxplot
sns.boxplot(x='product_name', y='rating', data=df, ax=ax2, palette=boxplot_palette)
# sns.boxplot(x='product_name', y='rating', data=df, ax=ax2, palette='pastel')
ax2.set_title('Star Rating Distribution by Product')
ax2.set_xlabel('Product Name')
ax2.set_ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig2)

# Average Sentiment Score per Product
st.subheader("🏷️ Average Sentiment Score per Product")
avg_sentiment = df.groupby('product_name')['sentiment_scores'].mean().sort_values(ascending=False)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=avg_sentiment.values, y=avg_sentiment.index, palette='YlOrBr', ax=ax3)
ax3.set_title('Average Sentiment Score per Product')
ax3.set_xlabel('Average Sentiment Score')
ax3.set_ylabel('Product Name')
ax3.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
st.pyplot(fig3)

# Filter ratings 1, 2, 3
st.subheader("⚠️ Sentiment Breakdown for Lower Ratings (1 to 3 Stars)")

low_rated_df = df[df['rating'].isin([1, 2, 3])]

# Show dataframe
st.write("Here's what customers are saying when they give 1, 2, or 3 stars:")
st.dataframe(low_rated_df[['product_name', 'rating', 'review_body', 'sentiment_label']])

# Count sentiment labels within low ratings
low_rating_sentiment = low_rated_df.groupby(['product_name', 'sentiment_label']).size().reset_index(name='count')

# Bar chart
fig_low = px.bar(low_rating_sentiment, 
                 x='product_name', 
                 y='count', 
                 color='sentiment_label',
                 color_discrete_map=mokobara_palette,
                 title="Sentiment Distribution for Ratings 1 to 3")
fig_low.update_layout(font=dict(family="Inter", size=14), xaxis_tickangle=-45)
st.plotly_chart(fig_low, use_container_width=True)


# Average Sentiment Score per Product for Negative Reviews
st.subheader("🚨 Average Sentiment Score per Product (Negative Reviews Only)")

# Filter only negative reviews (after corrected labeling)
negative_df = df[df['sentiment_label'] == 'Negative']

# Group and calculate average sentiment
avg_negative_sentiment = negative_df.groupby('product_name')['sentiment_scores'].mean().sort_values()

# Plot bar chart
fig_neg, ax_neg = plt.subplots(figsize=(10, 6))
sns.barplot(x=avg_negative_sentiment.values, y=avg_negative_sentiment.index, palette='Reds_r', ax=ax_neg)
ax_neg.set_title('Average Sentiment Score per Product (Negative Reviews)')
ax_neg.set_xlabel('Average Sentiment Score')
ax_neg.set_ylabel('Product Name')
ax_neg.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
st.pyplot(fig_neg)

# Word Cloud for Negative Reviews
st.subheader("☁️ Word Cloud — Negative Reviews Only")

# Generate text from only negative review bodies
negative_text = " ".join(negative_df['review_body'].dropna())

# Define a red/black color palette
negative_wordcloud_colors = ['#8B0000', '#FF0000', '#B22222', '#DC143C', '#2B2B2B']

def negative_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(negative_wordcloud_colors)

# Generate word cloud
negative_wordcloud = WordCloud(width=800, height=400, background_color='white',
                               color_func=negative_color_func).generate(negative_text)

# Show the word cloud
st.image(negative_wordcloud.to_array(), use_container_width=True)


# Filter & Explore Reviews
st.subheader("🔍 Explore Individual Reviews")
selected_product = st.selectbox("Select a product", sorted(df['product_name'].unique()))
filtered_df = df[df['product_name'] == selected_product]
st.write(f"Showing {len(filtered_df)} reviews for **{selected_product}**")
st.dataframe(filtered_df[['review_title', 'review_body', 'rating', 'sentiment_label']])
