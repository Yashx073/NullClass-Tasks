from utils.sentiment import detect_sentiment

def generate_response(user_input, context_response=""):
    """
    Generates a chatbot response based on user input and sentiment.
    Optional: context_response for additional info.
    """
    sentiment = detect_sentiment(user_input)

    if sentiment == "positive":
        prefix = "😊 I'm glad you feel that way! "
    elif sentiment == "negative":
        prefix = "😔 I'm sorry to hear that. "
    else:
        prefix = "🤖 Here's what I found: "

    # Combine with optional context
    return prefix + context_response
