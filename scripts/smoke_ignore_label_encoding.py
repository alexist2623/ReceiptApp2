import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.layoutlmv3_training import apply_word_ignores_to_encoding, labels_to_ids_with_ignore


class FakeEncoding(dict):
    def __init__(self, word_ids, labels):
        super().__init__()
        self._word_ids = word_ids
        self["labels"] = labels

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


def main():
    label2id = {"O": 0, "B-ITEM_NAME": 1, "B-ITEM_PRICE": 2}
    labels = ["B-ITEM_NAME", "IGNORE", "B-ITEM_PRICE"]
    label_ids, ignored = labels_to_ids_with_ignore(labels, label2id)
    encoding = FakeEncoding(
        word_ids=[[None, 0, 1, 1, 2, None]],
        labels=torch.tensor([[-100, label_ids[0], label_ids[1], label_ids[1], label_ids[2], -100]]),
    )
    apply_word_ignores_to_encoding(encoding, [ignored])
    expected = torch.tensor([[-100, 1, -100, -100, 2, -100]])
    passed = torch.equal(encoding["labels"], expected)
    report = {
        "input_labels": labels,
        "label_ids": label_ids,
        "ignored_word_indices": ignored,
        "output_token_labels": encoding["labels"].tolist(),
        "expected": expected.tolist(),
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
