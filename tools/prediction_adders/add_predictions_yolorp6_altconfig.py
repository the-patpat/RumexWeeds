import fiftyone as fo

#Load the dataset
dataset = fo.load_dataset('RumexWeeds')

classes = ['rumex_obtusifolius']
conf="025"
nms="045"
inf_size="1920"
#Add the predictions
#They're located in files under predictions/yolor/
with fo.ProgressBar() as pb:
    for sample in pb(dataset):

        filename = sample.filename.replace('png', 'txt')
        tags = sample.tags
        if "val" in sample.tags:
            split = "val"
        elif "test" in sample.tags:
            split = "test"
        else:
            continue
        try:
            with open(f'/home/pat/yolor/inference/Output_eval_rumex_yolor_p6_{split}_conf{conf}_nms{nms}/{filename}') as f:
                detections = []
                predictions = f.readlines()
                for prediction in predictions:
                    prediction = prediction.split(' ')
                    #Don't add every prediction (ran the detection with conf 0.0, so absolutely no NMS or anything)
                    if float(prediction[5]) > 0.005:
                        #xywh in file, where xy are CENTER coordinates, not top left, top right
                        prediction[1] = float(prediction[1]) - 0.5*float(prediction[3])
                        prediction[2] = float(prediction[2]) -  0.5*float(prediction[4])
                        detections.append(
                            fo.Detection(
                                label=classes[int(prediction[0])],
                                bounding_box=[float(x) for x in prediction[1:5]],
                                confidence=float(prediction[5])
                            )
                        )

                # Save predictions to dataset
                sample[f"predictions_yolor_p6_{inf_size}_{conf}_{nms}"] = fo.Detections(detections=detections)
                sample.save()
        except FileNotFoundError:
            pass