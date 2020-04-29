import detectron2
from extracting import img_
from extract_from_image import extract_from_images
import flask
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import request, jsonify, render_template
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np

from PIL import Image
import math
import os

from pdf2image import convert_from_path
from neo4j import GraphDatabase

def sendToNeo4j(query, **kwargs):
    print("saving to db")
    driver = GraphDatabase.driver("bolt://52.152.245.152:7687", auth=('neo4j', 'graph'))
    db = driver.session()
    consumer = db.run(query, **kwargs).consume()
    print("data saved")

 

app = flask.Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/uploads/'
ALLOWED_EXTENSIONS = set(['pdf'])
DEBUG = True

app.config.from_object(__name__)
app.config['SECRET_KEY'] = "K\x98\xa3\x89s\x146=\xe5\x97s\x17"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/api", methods=["POST"])
def process_score_image_request():
    if request.method == "POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
            pages = convert_from_path("/uploads/" + filename, dpi=200, fmt='jpeg')
            
            for idx, p in enumerate(pages):
              print(idx)
            # pages 
              im = np.array(p)[:, :, ::-1]
              predictions = default_predictor(im)
              instances = predictions["instances"].to('cpu')
              MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes = ['text', 'title', 'list', 'table', 'figure']
              pred_classes = instances.pred_classes



              labels = [classes[i] for i in pred_classes]
              label_count = [{i: labels.count(i)} for i in labels]
              label_count = [dict(y) for y in set(tuple(x.items()) for x in label_count)]
              # print(label_count)
              label_count = [{k:[v, []]} for label in label_count for k,v in label.items()]
              page_label_count = {f"page {idx}": label_count}
              # print(page_label_count)
              # print(label_count)  



              def add_content(content):
                for i in label_count:
                  for k, v in i.items():
                    if k == label:
                      v[1].append(content)
                return True



              boxes = instances.pred_boxes
              if isinstance(boxes, detectron2.structures.boxes.Boxes):
                  boxes = boxes.tensor.numpy()
              else:
                  boxes = np.asarray(boxes)


              from PIL import Image
              import math
              table = []
              list_ = []
              text = []
              title = []
              # content = [table]


              for label, bbox in zip(labels, boxes):

                # getting prediction bboxes from model outputs

                  x2 = math.ceil(bbox[0])
                  x1 = math.ceil(bbox[1])
                  y2 = math.ceil(bbox[2])
                  y1 = math.ceil(bbox[3])
                  crop_img = im[x1:y1,x2:y2]
                  if len(crop_img) <= 8:
                    continue
                  
                  
                  if label == "table":
                    print(label)
                    # add_content(img_(crop_img[ : , : , -1]))
                  elif label == "list":
                    add_content(extract_from_images(crop_img))
                  elif label == "title":
                    add_content(extract_from_images(crop_img))
                  elif label != "figure":
                    add_content(extract_from_images(crop_img))
                    # print(page_label_count)
              print(page_label_count)
              for k, v in page_label_count.items():
                # sendToNeo4j("MERGE (d:Document)-[:Page]->(p: Page {page_num: $k})", k=k)
                for i in v:
                  for l, m in i.items():
                    # print(m)
                    if l == 'figure':
                      sendToNeo4j("MERGE (d:Document) MERGE(d)-[:Page]->(p: Page {page_num: $page}) MERGE(p)-[:Figure_count {figure: $m}]->(f:Figure {figure: 'figure'})", m=m[0], page=k)
                    if l == 'text':
                      sendToNeo4j("MERGE (d:Document) MERGE(d)-[:Page]->(p: Page {page_num: $page}) MERGE(p)-[:Paragraph_count {text: $m}]->(pa:Paragraph {text: $text})", m=m[0], page=k, text=m[1])
                    if l == 'title':
                      sendToNeo4j("MERGE (d:Document) MERGE(d)-[:Page]->(p: Page {page_num: $page}) MERGE(p)-[:Title_count {title: $m}]->(t:Title {title: $title})", m=m[0], page=k, title=m[1])
                    if l == 'table':
                      sendToNeo4j("MERGE (d:Document) MERGE(d)-[:Page]->(p: Page {page_num: $page}) MERGE(p)-[:Table_count {table: $m}]->(ta:Table {table: $table})", m=m[0], page=k, table=m[1])
                    if l == 'form':
                       sendToNeo4j("MERGE (d:Document) MERGE(d)-[:Page]->(p: Page {page_num: $page}) MERGE(p)-[:Form_count {form: $m}]->(fo:Form {form: $form})", m=m[0], page=k, form=m[1])
            
                          # sendToNeo4j('MERGE(p:Page{page:$page_label_count.keys()[0]', keys=page_label_count.keys()[0])

       
        return render_template('index.html')

app.run(host="0.0.0.0", port=8899, debug=True)