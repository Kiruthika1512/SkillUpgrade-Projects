import spacy
from spacy.matcher import Matcher

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Define patterns for matching
patterns = [
    [{"LOWER": "hello"}],
    [{"LOWER": "hi"}],
    [{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}],
    [{"LOWER": "bye"}],
]

# Add patterns to the matcher
for pattern in patterns:
    matcher.add("Greeting", [pattern])

# Define responses
responses = {
    "hello": "Hi there!",
    "hi": "Hello!",
    "how are you": "I'm just a bot, but I'm doing well!",
    "bye": "Goodbye!",
}

# Function to get response
def get_response(text):
    doc = nlp(text)
    matches = matcher(doc)
    if matches:
        match_id, start, end = matches[0]
        span = doc[start:end]
        return responses.get(span.text.lower(), "Sorry, I didn't understand that.")
    else:
        return "Sorry, I didn't understand that."

# Test the chatbot
queries = ["Hello", "How are you?", "Bye"]
for query in queries:
    response = get_response(query)
    print(f'User: {query}\nBot: {response}')
