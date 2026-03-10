def train_model(model, dataset_yaml,
                epochs=100,
                batch=8,
                imgsz=512,
                device="cpu"):
    
    """
    Train YOLO mosquito detection model.
    """

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=15,
        device=device,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1
    )

    return results
