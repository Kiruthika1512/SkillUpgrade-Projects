from textblob import TextBlob
import pandas as pd

# Load a sample dataset (e.g., movie reviews)
data = {'review': ["I love this movie", "I hate this movie", "It was okay, not great but not terrible either"]}
df = pd.DataFrame(data)

# Function to calculate sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis
df['sentiment'] = df['review'].apply(get_sentiment)

# Report the overall sentiment distribution
sentiment_counts = df['sentiment'].value_counts(bins=3)
print(sentiment_counts)

# Display the DataFrame with sentiment scores
print(df)
