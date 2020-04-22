import flask
from flask_cors import CORS
from flask import request, jsonify
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np
from extracting import img_
from extract_from_images import extract_from_images
from PIL import Image
import math



def prepare_predictor():
    data = request.files['file']
    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfgFile = "DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "model_final_trimmed.pth"
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy
    # boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()[0]
    classes = ['text', 'title', 'list', 'table', 'figure']
    default_predictor = detectron2.engine.defaults.DefaultPredictor(cfg)
    img = detectron2.data.detection_utils.read_image(data, format="BGR")
    print("Predictor has been initialized.")
    predictions = default_predictor(img)
    instances = predictions["instances"].to('cpu')
    
    return img, instances, classes

app = flask.Flask(__name__)
CORS(app)
img, instances, classes = prepare_predictor()


@app.route("/api", methods=["POST"])
def process_score_image_request():
    pred_classes = instances.pred_classes
    labels = [classes[i] for i in pred_classes]
    # print(labels)
    boxes = instances.pred_boxes
    if isinstance(boxes, detectron2.structures.boxes.Boxes):
        boxes = boxes.tensor.numpy()
    else:
        boxes = np.asarray(boxes)

    for label, bbox in zip(labels, boxes):
         
        # getting prediction bboxes from model outputs
        
        x2 = math.ceil(bbox[0])
        x1 = math.ceil(bbox[1])
        y2 = math.ceil(bbox[2])
        y1 = math.ceil(bbox[3])
        crop_img = img[x1:y1,x2:y2]
        print(len(crop_img))
        if len(crop_img) <= 8:
          continue
        if label == "table":
          table_ = img_(crop_img[ : , : , -1])
          print("----------------")
          print(label)
          print("----------------")
          print(table_.head(10))
        elif label != "figure":
          print("----------------")
          print(label)
          print("----------------")
          print(extract_from_images(crop_img))

    return 'done'

app.run(host="0.0.0.0", port=8899)
