from ultralytics import YOLO
model = YOLO('yolov8x')
results = model.predict('input_videos/image.png',save=True)
print(results)
