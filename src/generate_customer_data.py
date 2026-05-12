from pathlib import Path

from segmentation_utils import save_customer_dataset


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "customer_segmentation_enhanced.csv"
    save_customer_dataset(output, n_customers=360, random_state=42)
    print(f"Saved dataset to {output}")
