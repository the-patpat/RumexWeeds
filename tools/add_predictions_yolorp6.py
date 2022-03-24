import fiftyone as fo
from sklearn.model_selection import train_test_split

#Load the dataset
dataset = fo.load_dataset('RumexWeeds')

classes = ['rumex_obtusifolius']
#Add the predictions
#They're located in files under predictions/yolor/
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
        with open(f'/home/pat/yolor/inference/Output_eval_rumex_yolor_p6_{split}/{filename}') as f:
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
            sample["predictions_yolor_p6_1920"] = fo.Detections(detections=detections)
            sample.save()