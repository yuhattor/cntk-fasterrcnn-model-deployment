from __future__ import print_function
import numpy as np
import os, sys
import yaml  
import cntk
from cntk import load_model, Axis, input_variable, parameter, times, roipooling
from cntk.core import Value
from cntk.io import MinibatchData
from cntk.layers import Constant
from _cntk_py import force_deterministic_algorithms

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.rpn.rpn_helpers import create_rpn, create_proposal_target_layer
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss
from utils.map.map_helpers import evaluate_detections
from utils.annotations.annotations_helper import parse_class_map_file
from config import cfg
from od_mb_source import ObjectDetectionMinibatchSource
from cntk_helpers import regress_rois
#from collections import OrderedDict
from flask import Flask, request, jsonify
import urllib.request as ur
import pprint



###############################################################
image_width = cfg["CNTK"].IMAGE_WIDTH
image_height = cfg["CNTK"].IMAGE_HEIGHT
num_channels = 3

# dims_input -- (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
dims_input_const = MinibatchData(Value(batch=np.asarray(
    [image_width, image_height, image_width, image_height, image_width, image_height], dtype=np.float32)), 1, 1, False)

# Color used for padding and normalization (Caffe model uses [102.98010, 115.94650, 122.77170])
img_pad_value = [103, 116, 123] if cfg["CNTK"].BASE_MODEL == "VGG16" else [114, 114, 114]
normalization_const = Constant([[[103]], [[116]], [[123]]]) if cfg["CNTK"].BASE_MODEL == "VGG16" else Constant([[[114]], [[114]], [[114]]])

# Compute mean average precision on test set
map_file_path = os.path.join(abs_path, cfg["CNTK"].MAP_FILE_PATH)
data_path = map_file_path

globalvars = {}
globalvars['class_map_file'] = os.path.join(data_path, cfg["CNTK"].CLASS_MAP_FILE)
globalvars['classes'] = parse_class_map_file(globalvars['class_map_file'])
globalvars['num_classes'] = len(globalvars['classes'])

LabelList =  globalvars['classes'] 
model_path = "faster.model"

if cfg["CNTK"].FORCE_DETERMINISTIC:
    force_deterministic_algorithms()

# model specific parameters
feature_node_name = cfg["CNTK"].FEATURE_NODE_NAME
last_hidden_node_name = cfg["CNTK"].LAST_HIDDEN_NODE_NAME
roi_dim = cfg["CNTK"].ROI_DIM



###############################################################
# The main method trains and evaluates a Fast R-CNN model.
# If a trained model is already available it is loaded an no training will be performed (if MAKE_MODE=True).
#if __name__ == '__main__':
def evalImage(url):
    # set image
    eval_model = load_model(model_path)

    classes = globalvars['classes']
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    frcn_eval = eval_model(image_input, dims_input)
    
    # Create the minibatch source
    
    minibatch_source = ObjectDetectionMinibatchSource(
        url,
        max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
        pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
        randomize=False, use_flipping=False,
        max_images=cfg["CNTK"].NUM_TEST_IMAGES)
    
    # define mapping from reader streams to network inputs
    input_map = {
        minibatch_source.image_si: image_input,
        minibatch_source.roi_si: roi_input,
        minibatch_source.dims_si: dims_input
    }
    
    # evaluate test images and write netwrok output to file
    all_gt_infos = {key: [] for key in classes}
    img_i = 0
    mb_data = minibatch_source.next_minibatch(url, 1, input_map=input_map)

    gt_row = mb_data[roi_input].asarray()
    gt_row = gt_row.reshape((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5))
    all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]
    
    for cls_index, cls_name in enumerate(classes):
        if cls_index == 0: continue
        cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
        all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                       'difficult': [False] * len(cls_gt_boxes),
                                       'det': [False] * len(cls_gt_boxes)})
    
    output = frcn_eval.eval({image_input: mb_data[image_input], dims_input: mb_data[dims_input]})

    out_dict = dict([(k.name, k) for k in output])
    out_cls_pred = output[out_dict['cls_pred']][0]
    out_rpn_rois = output[out_dict['rpn_rois']][0]
    out_bbox_regr = output[out_dict['bbox_regr']][0]
    
    labels = out_cls_pred.argmax(axis=1)
    scores = out_cls_pred.max(axis=1)

    result = dict()
    for label in LabelList:
        result.update({label:0})

    for index, label in enumerate(labels):
        if result[LabelList[int(label)]] < scores[index]:
          result.update({LabelList[int(label)]:scores[index]})

    pp = pprint.PrettyPrinter(indent=4)
    print("---------------------")
    print(url)
    pp.pprint(result)
    print("---------------------")

    for number, accuracy in result.items():
        result.update({number:str(accuracy)})
    return result


app = Flask(__name__)
@app.route("/", methods=['POST'])

def index():
    content = request.json
    print(content["url"])
    return jsonify(evalImage(content["url"]))

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)