import cv2, time, threading, os, signal, sys
from flask import Flask, Response
from ultralytics import YOLO

model = YOLO('/home/student/Desktop/best.pt')
app = Flask(__name__)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
last_frame = [None]
os.makedirs('/home/student/Desktop/snapshots', exist_ok=True)

def gen():
    while True:
        ret, frame = cap.read()
        if not ret: continue
        results = model.predict(source=frame, conf=0.4, imgsz=320, verbose=False, device='cpu')
        annotated = results[0].plot()
        last_frame[0] = annotated.copy()
        _, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'

def keyboard_thread():
    print("[INFO] Apasa 'S' + Enter pentru snapshot | Ctrl+C pentru oprire")
    while True:
        try:
            key = input().strip().lower()
            if key == 's':
                if last_frame[0] is not None:
                    filename = f"/home/student/Desktop/snapshots/snap_{int(time.time())}.jpg"
                    cv2.imwrite(filename, last_frame[0])
                    print(f"[SNAPSHOT] Salvat: {filename}")
                else:
                    print("[WARN] Niciun frame disponibil inca!")
        except EOFError:
            break

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

t = threading.Thread(target=keyboard_thread, daemon=True)
t.start()

try:
    app.run(host='0.0.0.0', port=5000)
except KeyboardInterrupt:
    print("\n[INFO] Oprire...")
    cap.release()
    sys.exit(0)
