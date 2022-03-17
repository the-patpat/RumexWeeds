#%% Load dataset
import fiftyone as fo
import os
import json
import glob
import torch 
import sys
from PIL import Image
from torchvision.transforms import functional as func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_name = "RumexWeeds"
dataset_top_dir = "../data/"
data_path = "imgs/"
labels_path = "annotations.xml"

assert fo.dataset_exists(base_name), "RumexWeeds dataset does not exist!"

dataset = fo.load_dataset(base_name)

#%% Add yolov5-l prediction
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