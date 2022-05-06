#%% Load dataset
import fiftyone as fo
import os
import json
import glob
import numpy as np
import torch 
import copy
import sys
from PIL import Image
from torchvision.transforms import functional as func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir('/mnt/d/OneDrive - Danmarks Tekniske Universitet/Thesis/Experiments/RumexWeeds')

base_name = "RumexWeeds"
dataset_top_dir = os.path.abspath('data/')
data_path = "imgs/"
labels_path = "annotations.xml"

assert fo.dataset_exists(base_name), "RumexWeeds dataset does not exist!"

dataset = fo.load_dataset(base_name)

#%% Add yolov5-l prediction (GPU, local forward pass method)
def add_preds_yolov5(dataset, classes,  model, field_name='predictions'):
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            if sample.has_field(field_name):
                if sample[field_name] is not None:
                    continue
            image = Image.open(sample.filepath)
            image = func.to_tensor(image).to(device)
            image = image.float()
            c, h, w = image.shape
            
            preds = model(sample.filepath)
            preds = preds.xyxyn[0]
            boxes = preds[:, 0:4].cpu().detach().numpy()

            labels = preds [:, 5].cpu().detach().numpy().astype(int)
            scores = (preds[:, 4]).cpu().detach().numpy()
    #preds = model(image.half())[0]

            # Convert detections to FiftyOne format
            detections = []
            for label, score, box in zip(labels, scores, boxes):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1, y1, (x2 - x1), (y2 - y1)]

                detections.append(
                    fo.Detection(
                        label=classes[label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

            # Save predictions to dataset
            sample[field_name] = fo.Detections(detections=detections)
            sample.save()


#%% Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/mnt/d/OneDrive - Danmarks Tekniske Universitet/Thesis/Experiments/RumexWeeds-YOLOv5/weights/rumex_obs_l/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

#%%
add_preds_yolov5(dataset, ["rumex_obtusifolius"], model, 'predictions_yolov5_l')

#%% json-based method 
def add_preds_from_json(dataset : fo.Dataset, path_to_preds: str, field_name=f"predictions", classes=None, score_multiplicator=1):
    """Adds predictions from coco format json file
    
    Parameters
    -----------
    dataset: FiftyOne Dataset or DatasetView
        Contains the samples for which the predictions will be added
        Path to the coco-json file that contains the ground_truth_annotations / image_id - filepath mapping
    path_to_preds: filepath, str
        Path to the coco-json file that contains the predictions from the detector, identified by anno id and image id
    field_name : str
        Name of the field in which the detections will be stored
    classes : list
        For future use, not implemented yet
    """
    #Load the predictions
    with open(path_to_preds, 'r') as f:
        preds = json.load(f)
        if classes is None:
            classes = [x["name"] for x in preds["categories"]]

    with fo.ProgressBar() as pb:
        for pred in pb(preds):

            image_id = pred["image_id"]

            #Retrieve image metadata
            #Load image from dataset
            sample_location = image_id.split('_')[0] + '_'+ image_id.split('_')[1]
            sequence_num = 'seq' + image_id.split('_')[-2]
            sample = dataset[os.path.abspath(os.path.join(dataset_top_dir, sample_location, sequence_num, 'imgs', image_id)) + '.png']
            
            #Load the predicted image bbox
            bbox = pred["bbox"]
            bbox_orig = copy.deepcopy(bbox)

            assert (np.asarray(bbox) >= 0.0).all(), f"Something has gone wrong, original bbox: {bbox}"

            #Normalize
            bbox[0] /= sample["metadata"]["width"]
            bbox[1] /= sample["metadata"]["height"]
            bbox[2] /=  sample["metadata"]["width"]
            bbox[3] /= sample["metadata"]["height"]

            assert (np.asarray(bbox) >= 0.0).all(), f"Something has gone wrong, original bbox: {bbox_orig}"
       

            label = pred["category_id"]

            if sample.has_field(field_name):
                if sample[field_name] is None:
                    sample[field_name] = fo.Detections()
                else:
                    detections = copy.deepcopy(sample[field_name]["detections"])
                    detections.append(fo.Detection(label=classes[label], bounding_box=bbox, confidence=(pred["score"]*score_multiplicator if "score" in pred else None)))
                    sample[field_name] = fo.Detections(detections=detections)
            else:
                sample[field_name] = fo.Detections()
            sample.save()

#%% Add val predictions
# SCORE HAS TO BE MULTIPLIED by 100!
add_preds_from_json(dataset, '/home/pat/galirumi-dtu-transfer/yolov5/runs/val/exp/best_predictions.json', field_name="predictions_yolov5_l_single", classes=["rumex"], score_multiplicator=100)
add_preds_from_json(dataset, '/home/pat/galirumi-dtu-transfer/yolov5/runs/test/exp/best_predictions.json', field_name="predictions_yolov5_l_single", classes=["rumex"], score_multiplicator=100)



#%% Manually correct the scores
for sample in dataset.match_tags(["val", "test"]):
    if sample.has_field("predictions_yolov5_l_single"):
        if sample["predictions_yolov5_l_single"] is not None:
            for prediction in sample["predictions_yolov5_l_single"]["detections"]:
                print(prediction.confidence)
                prediction.confidence *= 100
            sample.save()