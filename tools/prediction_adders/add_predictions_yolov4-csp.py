import fiftyone as fo
from sklearn.model_selection import train_test_split
import os

#Load the dataset
dataset = fo.load_dataset('RumexWeeds')

classes = ['rumex']
#Add the predictions
#They're located in files under predictions/yolor/
with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        filename = sample.filename.replace('png', 'txt')
        tags = sample.tags
        
        if os.path.exists(f'/home/pat/galirumi-dtu-transfer/ScaledYOLOv4/runs/test/exp/labels/{filename}'):
            with open(f'/home/pat/galirumi-dtu-transfer/ScaledYOLOv4/runs/test/exp/labels/{filename}') as f:
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
                sample["predictions_yolov4_csp_single"] = fo.Detections(detections=detections)
                sample.save()
        