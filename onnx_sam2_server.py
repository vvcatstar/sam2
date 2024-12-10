import yaml
import os 
import numpy as np
import supervision as sv
import time 
import cv2 

from copy import deepcopy
from supervision.draw.color import ColorPalette
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from onnx_sam2.sam2 import SAM2Image, draw_masks, draw_mask
from flask import Flask, request, jsonify
from IPython import embed
CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]
import random
def generate_random_rgb_color():
    r = random.randint(0, 255)  # 红色通道
    g = random.randint(0, 255)  # 绿色通道
    b = random.randint(0, 255)  # 蓝色通道
    return (r, g, b)

# 示例调用

class ONNXSAM2:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)['sam_server']
        encoder_model_path = self.configs['encoder_onnx']
        decoder_model_path = self.configs['decoder_onnx']
        self.model = SAM2Image(encoder_model_path, decoder_model_path)
        self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        
    def process_image(self, image_path, input_boxes, output_root):
        # input_boxes is numpy array
        image_name = image_path.split('/')[-1].split('.')[0]
        os.makedirs(output_root, exist_ok=True)
        output_result = os.path.join(output_root, image_name+'_sam2_mask')
        image = cv2.imread(image_path)
        self.model.set_image(image)
        input_boxes = np.array(input_boxes)[:, :4].astype('int')
        label_id = 1
        masks = []
        draw_image = deepcopy(image)
        for box in input_boxes:
            tuple_box = ((box[0], box[1]), (box[2], box[3]))   
            self.model.set_box(tuple_box, label_id)
            mask = self.model.get_masks()
            random_color = generate_random_rgb_color()
            
            draw_image = draw_mask(draw_image, mask[label_id], random_color, draw_border=True)
            masks.append(mask[label_id])
        masks = np.array(masks)
        np.save(output_result, masks)
        cv2.imwrite(output_result+'.jpg', draw_image)
        return output_result+'.npy'
    
sam2_config_file = '../config.yaml'
sam2 = ONNXSAM2(sam2_config_file)
# sam2.process_image(image_path='notebooks/images/cars.jpg', input_boxes=[[20, 20, 50, 50, 'car', '0.5'], [50, 50, 80, 80, 'car', '0.8']], output_root='./test')

app = Flask(__name__)
@app.route('/sam_segmentation', methods=['POST'])
def sam_segmentation():
    start_time = time.time()
    data = request.get_json()
    image_path = data['image_path']
    input_boxes = data['input_boxes']
    task = data['task']
    output_root = os.path.dirname(image_path)
    output_root = data.get('output_root', output_root)
    output_file = sam2.process_image(image_path=image_path, input_boxes=input_boxes, output_root=output_root)
    response = {}
    response['output_path'] = output_file
    end_time = time.time()
    use_time = round(end_time - start_time, 3)
    print(use_time)
    return jsonify(response)      
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10005)
            