from receipt_ocr_compare.normalization import normalize_text


def test_normalization_keeps_numeric_punctuation():
    assert normalize_text("  -1,234.50  ") == "-1,234.50"
    assert normalize_text("₩ 10,000") == "₩ 10,000"
    assert normalize_text("12:30") == "12:30"

