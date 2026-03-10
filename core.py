from ultralytics import YOLO
from .trainer import train_model
from .inference import run_inference
from .metrics import evaluate_model
from .dataset_validator import validate_dataset

class MosquitoYOLO:
    """
    Main library-style interface for mosquito detection.
    """

    def __init__(self, weights="yolo12n.pt"):

        self.model = YOLO(weights)

    def validate_dataset(self, dataset_yaml):
        """
        Validate dataset structure before training.
        """
        validate_dataset(dataset_yaml)

    def train(self, dataset_yaml, **kwargs):
        """
        Train mosquito detection model. 
        """
        train_model(self.model, dataset_yaml, **kwargs)

    def evaluate(self, dataset_yaml):
        """
        Evaluate model and compute metrics.
        """
        return evaluate_model(self.model, dataset_yaml)

    def infer(self, image_path):
        """
        Run inference on a single image.
        """
        return run_inference(self.model, image_path)
