
import json

from . import data_loader as dl
from .prediction_service import PredictionService


def main():
    db = dl.Database.from_env()
    print(json.dumps(PredictionService().generate_all(db), indent=2))


if __name__ == "__main__":
    main()
