import yaml
import os 
import numpy as np
import supervision as sv

from supervision.draw.color import ColorPalette
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from flask import Flask, request, jsonify
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
class SAM2:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        sam2_checkpoint = self.configs['sam2_checkpoint']
        model_cfg = self.configs["model_cfg"]
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        
    def process_image(self, image_path, input_boxes, output_path):
        # input_boxes is numpy array
        image_name = image_path.split('/')[-1].split('.')[0]
        output_root = os.path.join(output_path, image_name)
        os.makedirs(output_root, exist_ok=True)
        output_result = os.path.join(output_root, image_name+'_sam2_mask')
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        input_boxes = np.array(input_boxes)
        bboxes = input_boxes[:, :4].astype('int')
        # label = input_boxes[:, 4:]
        self.sam2_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bboxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        np.save(output_result, masks)
        return output_result+'.npy'
    
sam2_config_file = './sam_server_config.yaml'
sam2 = SAM2(sam2_config_file)
# sam2.process(image_path='notebooks/images/cars.jpg', input_boxes=[[20, 20, 50, 50, 'car', '0.5']], output_path='./test')
app = Flask(__name__)
@app.route('/sam_segmentation', methods=['POST'])
def sam_segmentation():
    data = request.get_json()
    image_path = data['image_path']
    input_boxes = data['input_boxes']
    task = data['task']
    output_root = os.path.dirname(image_path)
    output_root = data.get('output_root', output_root)
    output_file = sam2.process_image(image_path=image_path, input_boxes=input_boxes, output_path=output_root)
    response = {}
    response['output_path'] = output_file
    print(response)
    return jsonify(response)      
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10001)
            