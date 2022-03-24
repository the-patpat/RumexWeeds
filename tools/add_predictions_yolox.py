#%% Imports
import fiftyone as fo
import json
import os
import copy

dataset = fo.load_dataset('RumexWeeds')
dataset_top_dir = "../data/"
classes = dataset.default_classes
#%% YOLOx labels add (obtained via tools/eval.py on HPC) function
def add_preds_yolox_from_json(dataset, path_to_targets, path_to_preds, field_name=f"predictions", device='gpu'):
    #Load the ground truth that also contains the image id link
    with open(path_to_targets, 'r') as f:
        targets = json.load(f)
    
    #Load the predictions
    with open(path_to_preds, 'r') as f:
        preds = json.load(f)
    
    with fo.ProgressBar() as pb:
        for pred in pb(preds):

            image_id = int(pred["image_id"])

            img_data = targets["images"][image_id]

            #Check for equality
            assert int(img_data["id"]) == image_id, "IDs do not match"

            #Retrieve image metadata
            #Load image from dataset
            sample = dataset[os.path.abspath(os.path.join(dataset_top_dir, img_data["file_name"]))]
            
            #Load the predicted image bbox
            bbox = pred["bbox"]

            #Get the scaling factor (from ronjas code)
            s_w, s_h = img_data["width"]/sample["metadata"]["width"], img_data["height"]/sample["metadata"]["height"]

            #Bbox is xywh in absolute image coordinates (1920x1200)
            bbox[0] /= s_w
            bbox[1] /= s_h
            bbox[2] /= s_w
            bbox[3] /= s_h

            #Convert to normalized coordinates for FiftyOne
            bbox[0] /= sample["metadata"]["width"]
            bbox[1] /= sample["metadata"]["height"]
            bbox[2] /= sample["metadata"]["width"]
            bbox[3] /= sample["metadata"]["height"]
             
            label = pred["category_id"]

            if sample.has_field(field_name):
                if sample[field_name] is None:
                    sample[field_name] = fo.Detections()
                else:
                    detections = copy.deepcopy(sample[field_name]["detections"])
                    detections.append(fo.Detection(label=classes[label], bounding_box=bbox, confidence=pred["score"]))
                    sample[field_name] = fo.Detections(detections=detections)
            else:
                sample[field_name] = fo.Detections()
            sample.save()
#%% Add the predictions
add_preds_yolox_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds/cocoresults.json', field_name="predictions_yolox_DarkNet53_rumexweeds_json")
add_preds_yolox_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds/cocoresults.json', field_name="predictions_yolox_l_rumexweeds_json")
add_preds_yolox_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds/cocoresults.json', field_name="predictions_yolox_s_rumexweeds_json")
