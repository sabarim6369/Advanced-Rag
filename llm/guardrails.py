def check_query(query):
    blocked_words = ["ignore instructions", "give salary", "leak"]

    for word in blocked_words:
        if word in query.lower():
            return False
    return True