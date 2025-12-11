# config.py
import numpy as np


CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
LINE_POSITION = 0.6 
TARGET_CLASSES = [1, 2, 3, 5, 7] # bicycle, car, motorbike, bus, truck

# === CẤU HÌNH MODEL ===
MODEL_WEIGHTS = "yolov4-tiny.weights"
MODEL_CONFIG = "yolov4-tiny.cfg"
CLASS_FILE = "coco.names"

# === MÀU SẮC UI ===
COLOR_BG = "#2C3E50"
COLOR_ACCENT = "#1ABC9C"
COLOR_TEXT = "#ECF0F1"

# Load tên class
try:
    with open(CLASS_FILE, "r") as f:
        CLASSES = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
except FileNotFoundError:
    CLASSES = []
    COLORS = []