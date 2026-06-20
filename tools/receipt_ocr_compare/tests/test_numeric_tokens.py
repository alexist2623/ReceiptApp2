from receipt_ocr_compare.normalization import (
    corrected_numeric_text_optional,
    digit_sequence,
    is_numeric_token,
    normalize_text,
)


def test_numeric_token_preserves_decimal_comma_sign_and_currency():
    assert is_numeric_token("12.30")
    assert is_numeric_token("-1,234.50")
    assert is_numeric_token("₩10,000")
    assert is_numeric_token("$5.20")
    assert is_numeric_token("10%")
    assert is_numeric_token("****1234")


def test_normalized_text_does_not_apply_ocr_confusion_corrections():
    raw = "O.99"
    assert normalize_text(raw) == "O.99"
    assert corrected_numeric_text_optional(raw) == "0.99"


def test_digit_sequence_does_not_treat_confusable_letters_as_digits():
    assert digit_sequence("O.99") == "99"
    assert digit_sequence("12.30") == "1230"

