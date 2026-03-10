from mosquito_yolo import MosquitoYOLO

detector = MosquitoYOLO("yolo12n.pt")

detector.validate_dataset("data.yaml")

detector.train("data.yaml", epochs=100)

metrics = detector.evaluate("data.yaml")

detector.infer("test.jpg")
