import cv2
import numpy as np
from flask import Flask, jsonify, request
import threading
from flask_cors import CORS
import base64
from ultralytics import YOLO
import easyocr
from collections import Counter
import logging
from io import BytesIO

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Khởi tạo logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến global
previous_most_common = None
frame_with_plate = None
lock = threading.Lock()
valid_results = []

# Khởi tạo models
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# Bảng chuyển đổi ký tự
char_mapping = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '4': 'A'
}

def clean_plate_text(text):
    """Làm sạch văn bản: loại bỏ ký tự đặc biệt và khoảng trắng."""
    return ''.join(c for c in text if c.isalnum())

def fix_third_character(text):
    """Sửa ký tự thứ 3 nếu là số có thể nhầm lẫn với chữ cái."""
    if len(text) >= 3 and text[2] in char_mapping:
        text = text[:2] + char_mapping[text[2]] + text[3:]
    return text

def is_valid_plate(text):
    """Kiểm tra xem chuỗi có hợp lệ không."""
    if len(text) < 3:
        return False
    return text[2].isalpha()

def process_frame(frame):
    """Xử lý frame để cải thiện chất lượng nhận diện."""
    scale_factor = 2
    enlarged_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(enlarged_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    dilated = cv2.dilate(blurred, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff_image = 255 - cv2.absdiff(blurred, bg)
    return cv2.resize(diff_image, None, fx=2, fy=2, 
                     interpolation=cv2.INTER_CUBIC)

def process_image(image_data):
    """Xử lý ảnh và trả về biển số xe nếu tìm thấy."""
    global valid_results
    
    try:
        # Sử dụng YOLO để phát hiện biển số xe
        results = model.predict(source=image_data, conf=0.5, verbose=False)
        detections = results[0].boxes.xyxy

        for box in detections:
            x_min, y_min, x_max, y_max = map(int, box)
            plate_region = image_data[y_min:y_max, x_min:x_max]
            
            # Xử lý vùng chứa biển số
            processed_plate = process_frame(plate_region)
            
            # OCR trích xuất văn bản
            result = reader.readtext(processed_plate)
            
            if result:
                text = ' '.join([res[1] for res in result])
                cleaned_text = clean_plate_text(text)
                fixed_text = fix_third_character(cleaned_text)

                if is_valid_plate(fixed_text):
                    valid_results.append(fixed_text)
                    
                    # Kiểm tra nếu có đủ 7 kết quả hợp lệ
                    if len(valid_results) >= 7:
                        most_common = Counter(valid_results).most_common(1)[0][0]
                        
                        # Vẽ khung và text lên ảnh
                        cv2.rectangle(image_data, (x_min, y_min), 
                                    (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image_data, most_common, 
                                  (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                  (0, 255, 0), 2)
                        
                        # Chuyển ảnh sang base64
                        _, buffer = cv2.imencode('.jpg', image_data)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Reset kết quả
                        valid_results.clear()
                        
                        return most_common, img_base64

        return None, None

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

@app.route('/plate-number', methods=['POST'])
def get_most_common():
    global frame_with_plate
    try:
        # Nhận frame từ frontend
        frame_data = request.json.get("frame", "").split(',')[1]
        
        # Chuyển base64 thành ảnh
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Xử lý frame
        with lock:
            plate_number, processed_frame = process_image(frame)
            
            if plate_number and processed_frame:
                return jsonify({
                    'status': 'success',
                    'plate_number': plate_number,
                    'frame': processed_frame
                })
            else:
                return jsonify({
                    'status': 'processing',
                    'message': 'Processing frame...'
                })

    except Exception as e:
        logger.error(f"Error in get_most_common: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8089, threaded=True)
