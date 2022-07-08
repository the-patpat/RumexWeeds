#%%
import fiftyone as fo
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--fo_field", str, help="Name of FiftyOne field that will hold the predictions", default="predictions_yolor")
parser.add_argument("--preds_folder", str, help="Path to the prediction output folder")

opt = parser.parse_args()

#Load the dataset
dataset = fo.load_dataset('RumexWeeds')

classes = ['rumex']
#Add the predictions
#They're located in files under predictions/yolor/

def add_predictions_yolo(dataset, pred_directory, field_name, classes, include_train=True, split_directory=True):
    failed_files = []
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):

            filename = sample.filename.replace('png', 'txt')
            tags = sample.tags
            if "train" in sample.tags:
                split = "train"
            elif "val" in sample.tags:
                split = "val"
            elif "test" in sample.tags:
                split = "test"
            
            #If no train predictions are available skip this
            if not (split == "train" and not include_train): 
                if os.path.isfile(f'{pred_directory}/' + (f'{split}/' if split_directory else '') + filename):
                    with open(f'{pred_directory}/' + (f'{split}/' if split_directory else '') + filename) as f:
                        detections = []
                        predictions = f.readlines()
                        for prediction in predictions:
                            prediction = prediction.split(' ')
                            #Don't add every prediction (ran the detection with conf 0.0, so absolutely no NMS or anything)
                            if float(prediction[5]) > 0.005:

                                #xywh in file, where xy are CENTER coordinates, not top left, top right
                                prediction[1] = float(prediction[1]) - 0.5*float(prediction[3])
                                prediction[2] = float(prediction[2]) - 0.5*float(prediction[4])
                                detections.append(
                                    fo.Detection(
                                        label=classes[int(prediction[0])],
                                        bounding_box=[float(x) for x in prediction[1:5]],
                                        confidence=float(prediction[5])
                                    )
                                )

                        # Save predictions to dataset
                        sample[field_name] = fo.Detections(detections=detections)
                        sample.save()
                else:
                    sample[field_name] = fo.Detections(detections=None) 
                    failed_files.append(filename)
                    sample.save()
    dataset.save()
    return failed_files

failed_files = add_predictions_yolo(dataset, opt.preds_folder, opt.fo_field, ["rumex"], include_train=False, split_directory=False)
