# detector.py
import cv2
import numpy as np
import config 

class YoloDetector:
    def __init__(self):
        print("[INFO] Đang tải Model YOLO...")
        self.net = cv2.dnn.readNet(config.MODEL_WEIGHTS, config.MODEL_CONFIG)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > config.CONFIDENCE_THRESHOLD and class_id in config.TARGET_CLASSES:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
        
        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                results.append({
                    "box": boxes[i], 
                    "class_id": class_ids[i],
                    "confidence": confidences[i]
                })
        return results