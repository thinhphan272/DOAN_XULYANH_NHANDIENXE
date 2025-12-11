import cv2
import numpy as np
from centroidtracker import CentroidTracker

# === CẤU HÌNH ===
VIDEO_PATH = "Thailand_traffic_1.mp4" 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
LINE_POSITION = 0.6  

try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("[ERROR] Không tìm thấy file coco.names!")
    exit()


np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))


TARGET_CLASSES = [1, 2, 3, 5, 7] 


print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


ct = CentroidTracker(maxDisappeared=40)


total_counts = {
    "Car": 0,
    "Motorbike": 0,
    "Heavy": 0 
}
object_id_list = [] 

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Không mở được video {VIDEO_PATH}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video kết thúc.")
        break
    

    frame = cv2.resize(frame, (960, 540))
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD and class_id in TARGET_CLASSES:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    
    rects = [] 
    final_class_ids = [] 

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            rects.append((x, y, x + w, y + h))
            final_class_ids.append(class_ids[i])
            
            
    objects = ct.update(rects)

  
    line_y = int(height * LINE_POSITION)
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    for (objectID, centroid) in objects.items():
        # Vẽ tâm và ID
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

      
        if (line_y - 15) < centroid[1] < (line_y + 15):
            if objectID not in object_id_list:
                
              
                detected_class_name = "Unknown"
                min_dist = 99999
                
                for i, box in enumerate(rects):
                   
                    box_cx = int((box[0] + box[2]) / 2)
                    box_cy = int((box[1] + box[3]) / 2)
                    
                   
                    dist = np.sqrt((box_cx - centroid[0])**2 + (box_cy - centroid[1])**2)
                    
                    
                    if dist < min_dist and dist < 80: 
                        min_dist = dist
                        c_id = final_class_ids[i]
                        detected_class_name = classes[c_id]

                
                if detected_class_name in ["motorbike", "bicycle"]:
                    total_counts["Motorbike"] += 1
                elif detected_class_name == "car":
                    total_counts["Car"] += 1
                elif detected_class_name in ["bus", "truck"]:
                    total_counts["Heavy"] += 1
                
               
                object_id_list.append(objectID)
                
                
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 4)

 
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 110), (0, 0, 0), -1)
    alpha = 0.6 
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Ghi thông số
    cv2.putText(frame, f"Motorbike: {total_counts['Motorbike']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Car:       {total_counts['Car']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Bus/Truck: {total_counts['Heavy']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, "ESC to Exit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Traffic Counter System - Final", frame)
    
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()