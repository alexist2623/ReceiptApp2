"""Angle-aware custom receipt end-to-end evaluation entry point.

This thin wrapper intentionally reuses ``eval_custom_layoutlmv3_span_relg_e2e``.
The shared implementation now detects angle-aware LayoutLMv3 checkpoints and
passes token-aligned angle features when available.
"""

from scripts.eval_custom_layoutlmv3_span_relg_e2e import main


if __name__ == "__main__":
    main()
