from collections import Counter

from receipt_ocr_compare.metrics import (
    cer,
    character_accuracy,
    confusion_pairs,
    decimal_point_mismatch_count,
    digit_sequence_exact_match,
    exact_match,
    punctuation_mismatch_count,
    sign_mismatch_count,
    thousands_separator_mismatch_count,
)


def test_exact_and_digit_sequence_metrics_keep_decimal_miss_as_error():
    assert not exact_match("12.30", "1230")
    assert digit_sequence_exact_match("12.30", "1230")
    assert decimal_point_mismatch_count("12.30", "1230") == 1


def test_cer_and_character_accuracy():
    assert cer("12.30", "12.30") == 0.0
    assert cer("12.30", "1230") == 0.2
    assert character_accuracy("12.30", "1230") == 0.8


def test_punctuation_confusion_and_sign_counts():
    assert punctuation_mismatch_count("-1,234.50", "1234,50") == 2
    assert thousands_separator_mismatch_count("1,234", "1234") == 1
    assert sign_mismatch_count("-10.00", "10.00") == 1
    assert confusion_pairs("O1S8.", "015B,") == Counter({"O/0": 1, "S/5": 1, "8/B": 1, "./,": 1})
