from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("../receipt_training_data2").resolve()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CORD-v2 from Hugging Face...")
    dataset = load_dataset("naver-clova-ix/cord-v2")

    print("Dataset loaded:")
    print(dataset)

    print(f"Saving dataset to: {OUT_DIR}")
    dataset.save_to_disk(str(OUT_DIR))

    print("Done.")
    print(f"Saved CORD-v2 dataset at: {OUT_DIR}")

if __name__ == "__main__":
    main()
