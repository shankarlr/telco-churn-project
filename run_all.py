from src.data_prep import run as run_prep
from src.features import run as run_features
from src.model import train_and_save

def main():
    print("1) Data cleaning")
    run_prep()
    print("2) Feature engineering")
    run_features()
    print("3) Train model")
    metrics = train_and_save()
    print("Done. Metrics:", metrics)

if __name__ == "__main__":
    main()
