def extract_preferences(text: str) -> dict:
    text_lower = text.lower()

    prefs = {
        "genre": None,
        "mood": None,
        "language": None,
        "family_friendly": None,
        "runtime": None,
        "similar_to": None
    }

    genres = ["action", "comedy", "sci-fi", "thriller", "romance", "horror", "drama", "animation"]
    for genre in genres:
        if genre in text_lower:
            prefs["genre"] = genre

    if "funny" in text_lower or "light" in text_lower:
        prefs["mood"] = "funny/light"
    elif "dark" in text_lower:
        prefs["mood"] = "dark"
    elif "emotional" in text_lower:
        prefs["mood"] = "emotional"
    elif "exciting" in text_lower:
        prefs["mood"] = "exciting"

    if "family" in text_lower or "kids" in text_lower:
        prefs["family_friendly"] = True

    if "short" in text_lower or "under 2 hours" in text_lower:
        prefs["runtime"] = "under 2 hours"

    if "hindi" in text_lower:
        prefs["language"] = "Hindi"
    elif "english" in text_lower:
        prefs["language"] = "English"
    elif "korean" in text_lower:
        prefs["language"] = "Korean"

    if "like " in text_lower:
        prefs["similar_to"] = text.strip()

    return prefs