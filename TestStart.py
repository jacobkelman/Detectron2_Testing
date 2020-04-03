# Install dependencies (clear cache with --no-cache-dir)
# git clone https://github.com/facebookresearch/detectron2
# pip3 install shapely
# pip3 install geos
# pip3 install cython pyyaml==5.1
# sudo install_name_tool -add_rpath /usr/lib /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/torch/_C.cpython-38-darwin.so #noqa: E501
# install certificates in python folder
# pip3 install -e detectron2
# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load File
file = "randomstreet2.jpg"
filepath = "/Users/jacob/Documents/ProgrammingTest/Images/" + file
im = cv2.imread(filepath)
# cv2.imshow("ImageWindow", im)
# cv2.waitKey()

# Fitting the Model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# Changes to run locally
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Visual Outputs
v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.6)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("ImageWindow", v.get_image())

while True:
    key = cv2.waitKey(0)
    if key in [27, ord('q'), ord('Q')]:
        cv2.destroyAllWindows()
