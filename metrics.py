def evaluate_model(model, dataset_yaml):

    """
    Evaluate trained model using YOLO validation.
    """

    metrics = model.val(data=dataset_yaml)

    results = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.p,
        "Recall": metrics.box.r
    }

    print("Evaluation Metrics")
    print("------------------")

    for k,v in results.items():
        print(f"{k}: {v:.4f}")

    return results
