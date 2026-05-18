def decision_engine(features, config):

    # Crack length rule
    if features["crack_length_mm"] > config["max_crack_length"]:
        return "FAIL"

    # Crack count rule
    if features["crack_count"] > config["max_crack_count"]:
        return "FAIL"

    # Dark area rule
    if features["dark_percent"] > config["max_dark_percent"]:
        return "FAIL"

    return "PASS"