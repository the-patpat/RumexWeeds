#%% Imports
import fiftyone as fo
import json
import os
import copy
from fiftyone import ViewExpression as F
import numpy as np


dataset = fo.load_dataset('RumexWeeds')
dataset_top_dir = "/mnt/d/OneDrive - Danmarks Tekniske Universitet/Thesis/Experiments/RumexWeeds/data"
#classes = dataset.default_classes
#%% YOLOx labels add (obtained via tools/eval.py on HPC) function
def add_preds_from_json(dataset : fo.Dataset, path_to_targets: str, path_to_preds: str, box_transform, field_name=f"predictions", classes=None):
    """Adds predictions from coco format json file
    
    Parameters
    -----------
    dataset: FiftyOne Dataset or DatasetView
        Contains the samples for which the predictions will be added
    path_to_targets: filepath, str
        Path to the coco-json file that contains the ground_truth_annotations / image_id - filepath mapping
    path_to_preds: filepath, str
        Path to the coco-json file that contains the predictions from the detector, identified by anno id and image id
    box_transform: function
        Function that takes box, scaling factors and image width/height as arguments to transform
        the source bbox format to FiftyOne bbox format
    field_name : str
        Name of the field in which the detections will be stored
    classes : list
        For future use, not implemented yet
    """
    #Load the ground truth that also contains the image id link
    with open(path_to_targets, 'r') as f:
        targets = json.load(f)
        id_filename_list = np.asarray([(target["id"], target["file_name"]) for target in targets["images"]])
        if classes is None:
            classes = [x["name"] for x in targets["categories"]]

        #Some lists start with 0, some with 1
        index_offset = (targets["images"][0]["id"] == 1)

    #Load the predictions
    with open(path_to_preds, 'r') as f:
        preds = np.asarray(json.load(f))
        id_pred_list = np.asarray([int(pred["image_id"]) for pred in preds])  

    
    #Sample based loop (more efficient)
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            
            #First, get the corresponding numerical id corresponding to the filepath of the sample
            ind_id_file = np.argwhere(id_filename_list[:, 1] == '/'.join(sample["filepath"].split('/')[8:]) )
            
            #Extract the numerical ID if match is found
            if ind_id_file.shape[0] > 0:
                
                num_id = int(id_filename_list[ind_id_file, 0])
                
                #Retrieve the predictions
                sample_preds = preds[id_pred_list == num_id]
                
                detections = []
                
                for pred in sample_preds:
                    #Load the predicted image bbox
                    bbox = pred["bbox"]

                    #Get the scaling factor (from ronjas code)
                    img_data = targets["images"][num_id - index_offset]
                    assert int(img_data["id"]) == num_id, "ID mismatch"
                    s_w, s_h = img_data["width"]/sample["metadata"]["width"], img_data["height"]/sample["metadata"]["height"]

                    bbox = box_transform(bbox, (s_w, s_h), (sample["metadata"]["width"], sample["metadata"]["height"]))
                    
                    label = pred["category_id"]

                    detections.append(fo.Detection(label=classes[label], bounding_box=bbox, confidence=(pred["score"] if "score" in pred else None)))
                
                sample[field_name] = fo.Detections(detections=detections)
            
            sample.save()



    # with fo.ProgressBar() as pb:
    #     for pred in pb(preds):

    #         image_id = int(pred["image_id"])

    #         img_data = targets["images"][image_id - index_offset]

    #         #Check for equality
    #         assert int(img_data["id"]) == image_id, "IDs do not match"

    #         #Retrieve image metadata
    #         #Load image from dataset
    #         if "seq" not in img_data["file_name"]:
    #             filename_split = img_data["file_name"].split('_')
    #             location = '_'.join(filename_split[0:2])
    #             sequence = 'seq' + filename_split[3]
    #             sample = dataset[os.path.abspath(os.path.join(dataset_top_dir, location, sequence, 'imgs', img_data["file_name"]))]
    #         else:
    #             sample = dataset[os.path.abspath(os.path.join(dataset_top_dir, img_data["file_name"]))]
            
    #         #Load the predicted image bbox
    #         bbox = pred["bbox"]

    #         #Get the scaling factor (from ronjas code)
    #         s_w, s_h = img_data["width"]/sample["metadata"]["width"], img_data["height"]/sample["metadata"]["height"]

    #         bbox = box_transform(bbox, (s_w, s_h), (sample["metadata"]["width"], sample["metadata"]["height"]))
             
    #         label = pred["category_id"]

    #         if sample.has_field(field_name):
    #             if sample[field_name] is None:
    #                 sample[field_name] = fo.Detections()
    #             else:
    #                 detections = copy.deepcopy(sample[field_name]["detections"])
    #                 detections.append(fo.Detection(label=classes[label], bounding_box=bbox, confidence=(pred["score"] if "score" in pred else None)))
    #                 sample[field_name] = fo.Detections(detections=detections)
    #         else:
    #             sample[field_name] = fo.Detections()
    #         sample.save()

def yolox_transform(box, scaling, img_dims):
    """ Transforms a bbox from the YOLOx framework to target FiftyOne format
    """
    #Bbox is xywh in absolute image coordinates (1920x1200)
    s_w = scaling[0]
    s_h = scaling[1]

    box[0] /= s_w
    box[1] /= s_h
    box[2] /= s_w
    box[3] /= s_h

    #Convert to normalized coordinates for FiftyOne
    box[0] /= img_dims[0] 
    box[1] /= img_dims[1]
    box[2] /= img_dims[0] 
    box[3] /= img_dims[1]

    return box

def detectron2_transform(box, scaling, img_dims):
    """Transforms a bbox from the detectron2 framework (xyxy abs) to FiftyOne format"""
    s_w = scaling[0]
    s_h = scaling[1]

    box[0] /= s_w
    box[1] /= s_h
    box[2] /= s_w
    box[3] /= s_h

    #x2 to width
    box[2] -= box[0]

    #y2 to height
    box[3] -= box[1]

    #Normalize
    box[0] /= img_dims[0] 
    box[1] /= img_dims[1]
    box[2] /= img_dims[0] 
    box[3] /= img_dims[1]

    return box

#%% Add the predictions (rumex_obtusifolius dataset with crispus neglected (negative examples))
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds/cocoresults.json', yolox_transform, field_name="predictions_yolox_DarkNet53_rumexweeds", classes=["rumex_obtusifolius"])
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds/cocoresults.json', yolox_transform, field_name="predictions_yolox_l_rumexweeds", classes=["rumex_obtusifolius"])
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds/cocoresults.json', yolox_transform, field_name="predictions_yolox_s_rumexweeds", classes=["rumex_obtusifolius"])

#%% Add predictions trained on single rumex class
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds_rumexsingleclass/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds_rumexsingleclass/cocoresults.json', yolox_transform, field_name="predictions_yolox_DarkNet53_rumexweeds_single")
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds_rumexsingleclass/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds_rumexsingleclass/cocoresults.json', yolox_transform, field_name="predictions_yolox_l_rumexweeds_single")
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds_rumexsingleclass/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds_rumexsingleclass/cocoresults.json', yolox_transform, field_name="predictions_yolox_s_rumexweeds_single")

#%% Add predictions trained on single rumex class with mosaic and aligned hyperparameters
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds_rumexsingleclass_with_mosaic/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds_rumexsingleclass_with_mosaic/cocoresults.json', yolox_transform, field_name="predictions_yolox_DarkNet53_rumexweeds_mosaic_single")
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds_rumexsingleclass_with_mosaic/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_l_rumexweeds_rumexsingleclass_with_mosaic/cocoresults.json', yolox_transform, field_name="predictions_yolox_l_rumexweeds_mosaic_single")
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds_rumexsingleclass_with_mosaic/coco.json',
                        '/home/pat/gbar_transfer/scratch/YOLOX/YOLOX_outputs/yolox_s_rumexweeds_rumexsingleclass_with_mosaic/cocoresults.json', yolox_transform, field_name="predictions_yolox_s_rumexweeds_mosaic_single")

#%%Detectron results
def make_relative_path(detectron_image_path, dataset_dir):
    with open(detectron_image_path, 'r') as f:
        coco_dict = json.load(f)
    for image in coco_dict["images"]:
        image["file_name"] = image["file_name"].replace(dataset_dir if dataset_dir[-1] == '/' else dataset_dir + '/', '')
    with open(os.path.splitext(detectron_image_path)[0]+"rel"+os.path.splitext(detectron_image_path)[1], 'w') as f:
        json.dump(coco_dict, f)

make_relative_path('/home/pat/gbar_transfer/scratch/RumexWeeds-github/tools/output/images_fasterrcnn.json', '/work1/s202616/RumexWeeds/')
add_preds_from_json(dataset, '/home/pat/gbar_transfer/scratch/RumexWeeds-github/tools/output/images_fasterrcnnrel.json',
                        '/home/pat/gbar_transfer/scratch/RumexWeeds-github/tools/output/predictions_fasterrcnn.json', detectron2_transform, field_name="predictions_faster_rcnn_single")
#%%YOLOv5 single class results
yolo5_transform = detectron2_transform


#%%Transform the bboxes to the right format (detectron2 uses xyxy)
def xyxy2xywh(box, image_width, image_height, normalize_result=True):
    """Transforms a xyxy absolute box into FiftyOne top left xy and width/height (normalized) format"""
    result = [None for _ in range(4)]
    result[0] = box[0] / image_width
    result[1] = box[1] / image_height
    result[2] = (box[2] - box[0]) / (image_width if normalize_result else 1)
    result[3] = (box[3] - box[1] ) / (image_height if normalize_result else 1)
    assert result[2] >= 0 or result[3] >= 0, "Conversion went wrong, result dimensions are negative"
    return result

with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        if sample.has_field('predictions_faster_rcnn_single'):
            if sample["predictions_faster_rcnn_single"] is not None:
                for det in sample['predictions_faster_rcnn_single']['detections']:
                    det.bounding_box = xyxy2xywh(det.bounding_box, sample['metadata']["width"], sample["metadata"]["height"])
                sample.save()
#%%Add the single class ground truth annotations
dataset.clone_sample_field('ground_truth_detections', 'ground_truth_detections_single')
with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        if sample["ground_truth_detections_single"] is not None:
            for detection in sample["ground_truth_detections_single"]["detections"]:
                detection["label"] = "rumex"
        sample.save()

#%% Change the single class labels to the rumex label (if they are different)
with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        for field_name in ("predictions_yolox_DarkNet53_rumexweeds_single","predictions_yolox_l_rumexweeds_single","predictions_yolox_s_rumexweeds_single"):
            if sample[field_name] is not None:
                for detection in sample[field_name]["detections"]:
                    detection["label"] = "rumex"
        sample.save()

#%% Add CenterNet results
add_preds_from_json(dataset, '/home/pat/galirumi-dtu-transfer/scratch/RumexWeeds-COCO/labels/test.json', '/home/pat/galirumi-dtu-transfer/CenterNet/exp/ctdet/coco_dla_1x/results_test.json', yolox_transform, "predictions_CenterNet_ctdet_coco-dla-1x_single")
# %%
