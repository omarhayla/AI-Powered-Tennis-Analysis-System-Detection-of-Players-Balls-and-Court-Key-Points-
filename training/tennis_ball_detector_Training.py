#!pip install ultralytics
#!pip install roboflow


from roboflow import Roboflow
rf = Roboflow(api_key="FdA6iySnSKp1KVMN9Cif")
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov5")

import shutil
shutil.move("tennis-ball-detection-6/train","tennis-ball-detection-6/tennis-ball-detection-6/train",)
shutil.move("tennis-ball-detection-6/test","tennis-ball-detection-6/tennis-ball-detection-6/test",)
shutil.move("tennis-ball-detection-6/valid","tennis-ball-detefction-6/tennis-ball-detection-6/valid",)

#!yolo task=detect mode=train model=yolov5l6u.pt data={dataset.location}/data.yaml epochs=100 imgsz=640