import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, font as tkfont
import os
import config

from detector import YoloDetector
from counter import VehicleCounter

def run_processing(video_path, root_window):
    # Khởi tạo modules
    try:
        detector = YoloDetector()
        counter = VehicleCounter()
    except Exception as e:
        messagebox.showerror("Lỗi Init", f"Lỗi khởi tạo: {str(e)}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở video!")
        return

    # Ẩn cửa sổ menu chính
    root_window.withdraw()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize về chuẩn HD để xử lý nhanh
        frame = cv2.resize(frame, (960, 540))

        # --- BƯỚC 1: DETECTION (Người 2) ---
        detections = detector.detect(frame)

        # --- BƯỚC 2: COUNTING (Người 3) ---
        frame, counts = counter.process(frame, detections)

        # --- BƯỚC 3: HIỂN THỊ (Người 1) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (240, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "THONG KE TRUC TIEP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Xe may: {counts['Motorbike']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"O to:   {counts['Car']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Xe tai: {counts['Heavy']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        cv2.putText(frame, "Nhan [ESC] de quay lai", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("He Thong Dem Xe AI", frame)
        
        # Nhấn ESC để thoát video, quay về menu
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Hiện lại menu chính và báo cáo kết quả
    root_window.deiconify()
    msg = f"Kết quả đếm:\n- Xe máy: {counts['Motorbike']}\n- Ô tô: {counts['Car']}\n- Xe tải/Buýt: {counts['Heavy']}"
    messagebox.showinfo("Hoàn tất", msg)

# --- CLASS GIAO DIỆN CHÍNH ---
class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI TRAFFIC MANAGER")
        self.root.geometry("600x500") # Tăng chiều cao để chứa đủ nút
        self.root.configure(bg=config.COLOR_BG)
        
        # 1. Header (Tiêu đề)
        tk.Label(root, text="🚗", font=("Arial", 60), bg=config.COLOR_BG).pack(pady=30)
        
        tk.Label(root, text="HỆ THỐNG ĐẾM XE", font=("Helvetica", 24, "bold"), 
                 bg=config.COLOR_BG, fg=config.COLOR_TEXT).pack()
        
        tk.Label(root, text="Powered by YOLOv4 & OpenCV", font=("Helvetica", 10), 
                 bg=config.COLOR_BG, fg="#95A5A6").pack(pady=5)

        # 2. Khu vực nút bấm
        btn_frame = tk.Frame(root, bg=config.COLOR_BG)
        btn_frame.pack(pady=30)

        # Nút Mở Video (Màu Xanh)
        btn_open = tk.Button(btn_frame, text="📂 MỞ VIDEO & CHẠY", 
                             bg=config.COLOR_ACCENT, fg="white",
                             font=("Helvetica", 12, "bold"), 
                             width=25, height=2,
                             relief="flat", cursor="hand2",
                             command=self.start)
        btn_open.pack(pady=10)

        # Nút Thoát (Màu Đỏ) - THÊM MỚI
        btn_exit = tk.Button(btn_frame, text="❌ THOÁT CHƯƠNG TRÌNH", 
                             bg="#E74C3C", fg="white", # Màu đỏ
                             font=("Helvetica", 12, "bold"), 
                             width=25, height=2,
                             relief="flat", cursor="hand2",
                             command=self.exit_app)
        btn_exit.pack(pady=10)

    def start(self):
        filename = filedialog.askopenfilename(
            title="Chọn Video Giao Thông",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")]
        )
        if filename:
            run_processing(filename, self.root)

    def exit_app(self):
        """Hàm xử lý khi bấm nút Thoát"""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn thoát ứng dụng?"):
            self.root.destroy()

if __name__ == "__main__":
    # Kiểm tra file cấu hình trước khi chạy
    if not config.CLASSES:
        messagebox.showerror("Lỗi Cấu Hình", "Thiếu file coco.names hoặc chưa load được Config!")
    else:
        root = tk.Tk()
        app = ModernApp(root)
        root.mainloop()