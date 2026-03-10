import os
import yaml

def validate_dataset(dataset_yaml):

    """
    Validate dataset configuration and directory structure.
    """

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError("Dataset YAML not found")

    with open(dataset_yaml) as f:
        data = yaml.safe_load(f)

    required_keys = ["train","val","names"]

    for key in required_keys:

        if key not in data:
            raise ValueError(f"Missing key in dataset yaml: {key}")

    if not os.path.exists(data["train"]):
        raise FileNotFoundError("Training folder not found")

    if not os.path.exists(data["val"]):
        raise FileNotFoundError("Validation folder not found")

    print("Dataset validation successful")
