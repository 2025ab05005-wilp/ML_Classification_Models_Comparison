import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from ml_utils import load_dataset, train_models


def main() -> None:
    data_path = root_dir / "data.csv"
    model_dir = root_dir / "model"

    df = load_dataset(str(data_path))
    _, _, metrics = train_models(df, str(model_dir))

    print("Training complete. Metrics:")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
