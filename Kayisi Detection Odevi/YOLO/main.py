from ultralytics import YOLO
import torch
from ultralytics import YOLO
from roboflow import Roboflow

print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())


# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

rf = Roboflow(api_key="adbV2srlK6jG1CO7n2J5")
project = rf.workspace("umutsefkansak").project("kayisi-detection")
version = project.version(1)
dataset = version.download("yolov11")

if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)
