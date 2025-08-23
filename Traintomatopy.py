from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

  
    model = YOLO(r"C:\veri arttirimi\yolo11n-seg.pt")  
    model.train(
        data=r"C:\veri arttirimi\domat 3.v3i.yolov11\data.yaml",
        epochs=20,
        batch=8,
        imgsz=640,
        device='cuda',
        task="segment" )