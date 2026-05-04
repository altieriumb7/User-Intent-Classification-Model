from src.preprocessing import clean_text


def test_clean_text_normalizes_urls_email_and_whitespace():
    text = "  Contact ME at User@Example.com!!! Visit https://example.com/help  "
    assert clean_text(text) == "contact me at email visit url"


def test_clean_text_handles_none():
    assert clean_text(None) == ""
