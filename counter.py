# counter.py
import cv2
import numpy as np
from centroidtracker import CentroidTracker
import config

class VehicleCounter:
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=40)
        self.counts = {"Car": 0, "Motorbike": 0, "Heavy": 0}
        self.object_id_list = []

    def process(self, frame, detections):
        height, width = frame.shape[:2]
        line_y = int(height * config.LINE_POSITION)
        
       
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        
        rects = []
        for det in detections:
            x, y, w, h = det["box"]
            rects.append((x, y, x + w, y + h))
            
            
            color = config.COLORS[det["class_id"]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

      
        objects = self.ct.update(rects)

       
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

           
            if (line_y - 15) < centroid[1] < (line_y + 15):
                if objectID not in self.object_id_list:
                   
                    detected_class_name = "Unknown"
                    min_dist = 99999
                    
                    for det in detections:
                        x, y, w, h = det["box"]
                        box_cx = int(x + w / 2)
                        box_cy = int(y + h / 2)
                        dist = np.sqrt((box_cx - centroid[0])**2 + (box_cy - centroid[1])**2)
                        
                        if dist < min_dist and dist < 100:
                            min_dist = dist
                            c_id = det["class_id"]
                            detected_class_name = config.CLASSES[c_id]

                    
                    if detected_class_name in ["motorbike", "bicycle"]:
                        self.counts["Motorbike"] += 1
                    elif detected_class_name == "car":
                        self.counts["Car"] += 1
                    elif detected_class_name in ["bus", "truck"]:
                        self.counts["Heavy"] += 1
                    
                    self.object_id_list.append(objectID)
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 4)
        
        return frame, self.counts