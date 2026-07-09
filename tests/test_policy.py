from src.cache.policy import adaptive_threshold, classify_ttl


def test_time_sensitive_gets_short_ttl():
    assert classify_ttl("What is the latest news today?") == 3600.0


def test_stable_fact_gets_long_ttl():
    assert classify_ttl("What is the boiling point of water?") == 7 * 86400.0


def test_default_ttl_for_neutral_prompt():
    assert classify_ttl("Rewrite this paragraph to be shorter.", default_ttl=500.0) == 500.0


def test_adaptive_thresholds():
    assert adaptive_threshold("classification") == 0.90
    assert adaptive_threshold("creative") == 0.99
    assert adaptive_threshold("qa") == 0.95
    assert adaptive_threshold("unknown", base=0.97) == 0.97
