import cv2
from matplotlib import pyplot as plt

def run_inference(model, image_path):

    """
    Run detection on an image and display results.
    """

    results = model(image_path)

    for r in results:

        img = cv2.imread(image_path)

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        names = model.names

        for box, score, cls in zip(boxes, scores, class_ids):

            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls]} {score:.2f}"

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(img,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(255,255,255),2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return results
