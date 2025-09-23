from textblob import TextBlob

def detect_sentiment(text):
    """
    Detects sentiment of the input text.
    Returns 'positive', 'negative', or 'neutral'.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
