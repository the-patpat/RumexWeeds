import fiftyone as fo
import os
import json
import glob
import copy
import argparse
from fiftyone import ViewField as F
import datetime as dt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, help="Path to the RumexWeeds dataset", default="/RumexWeeds")
opt = parser.parse_args()

base_name = "RumexWeeds"
dataset_top_dir = opt.dataset_dir
data_path = "imgs/"
labels_path = "annotations.xml"

#%% From the CVAT dataset, add it to fiftyone as the two-class dataset 

#Read the train/test/val split (random) for tagging the samples
d_split = {}
for x in ["test", "train", "val"]:
    with open(os.path.join(dataset_top_dir,f"dataset_splits/random_{x}.txt"), 'r') as f:
        #Take abspath, easier to find later
        d_split[x] = [os.path.abspath(os.path.join(dataset_top_dir , loc_seq_pic.replace('\n', ''))) for loc_seq_pic in f.readlines()]
        f.close()


if not fo.dataset_exists(base_name):
    # Create the dataset by creating the sub-datasets and appending them to a list
    # Iterate through every collection ({date}_{location}) and therein, iterate through every sequence
    datasets = []
    for collection in glob.glob(os.path.join(dataset_top_dir + '2021*')):
        for sub_dataset in glob.glob(os.path.join(collection + '/seq*')):
            collection_name, seq_name = os.path.split(sub_dataset)
            collection_name = os.path.split(collection_name)[-1]
            name = f"{base_name}_{collection_name}_{seq_name}"
            dataset_dir = sub_dataset 
            if not fo.dataset_exists(name):
                datasets.append(fo.Dataset.from_dir(
                    dataset_dir=dataset_dir,
                    dataset_type=fo.types.CVATImageDataset,
                    name=name,
                    data_path=data_path,
                    labels_path=labels_path, 
                    label_field="ground_truth_detections",
                    tags=f"{collection_name}_{seq_name}"
                ))
            else:
                datasets.append(fo.load_dataset(name))

    #Verify that it's not empty
    assert len(datasets) > 0 , "No datasets found"
    
    #Assemble the final dataset
    dataset = fo.Dataset(base_name)
    for sequence in datasets:
        dataset.add_samples(iter(sequence))
    
    dataset.default_classes = ["rumex_obtusifolius", "rumex_crispus", "rumex"]
    dataset.classes["ground_truth_detections"] = ["rumex_obtusifolius", "rumex_crispus"]
    dataset.save()
    view = dataset.view()

    
    #Make the dataset persistent so we don't have to assemble it next time
    dataset.persistent = True

    #Delete other datasets
    fo.delete_datasets('RumexWeeds_*', verbose=True)
else:
    dataset = fo.load_dataset(base_name)
    if not dataset.persistent:
        dataset.persistent = True

#%% Load GPS, IMU, odometry data, train/test/val category, timestamp, single class field
dataset.clone_sample_field('ground_truth_detections', 'ground_truth_detections_single')
with fo.ProgressBar() as pb:
    for sample in pb(dataset):
        if 'location' not in sample.field_names or sample['location'] is None:
            filepath, filename = os.path.split(sample.filepath)
            filename = os.path.splitext(filename)[0]
            with open(os.path.join(filepath + '/../gps.json', 'r'))as f:
                gps_dict = json.load(f)
                gps_data = gps_dict[filename.replace('_rgb', '')]
                sample['location'] = fo.GeoLocation(point=[gps_data['longitude'], gps_data['latitude']])
                print(f"Added GPS info to sample {filename}: {sample['location']}\r", end="")
                sample.save()
                f.close()
        if 'imu' not in sample.field_names or sample['imu'] is None:
            filepath, filename = os.path.split(sample.filepath)
            filename = os.path.splitext(filename)[0]
            with open(os.path.join(filepath + '/../imu.json', 'r'))as f:
                gps_dict = json.load(f)
                gps_data = gps_dict[filename.replace('_rgb', '')]
                sample['imu'] = gps_data 
                sample.save()
                f.close()
        if 'odom' not in sample.field_names or sample['odom'] is None:
            filepath, filename = os.path.split(sample.filepath)
            filename = os.path.splitext(filename)[0]
            with open(os.path.join(filepath + '/../odom.json', 'r')) as f:
                gps_dict = json.load(f)
                gps_data = gps_dict[filename.replace('_rgb', '')]
                sample['odom'] = gps_data 
                sample.save()
                f.close()
        for split in d_split:
            if os.path.abspath(sample.filepath) in d_split[split]:
                sample.tags.append(split)
                sample.tags = list(set(sample.tags))
                sample.save()
        
        #Create timestamp
        unixtime_str = os.path.splitext(os.path.split(sample.filepath)[-1])[0].split('_')[-1]
        sample['created_at'] = dt.datetime(*list(time.gmtime(float(unixtime_str)*1e-9)[0:6]))
        sample.save()
        
        #Single class detections field
        if sample["ground_truth_detections_single"] is not None:
            for detection in sample["ground_truth_detections_single"]["detections"]:
                detection["label"] = "rumex"
        sample.save()

#%%Create Summer tag (June-August)
summer_view = dataset.match(fo.ViewField("created_at").month() >= 6).match(fo.ViewField("created_at").month() <= 8)
summer_view.tag_samples("summer")

#%%Create Autumn tag
autumn_view = dataset.match(fo.ViewField("created_at").month() >= 9).match(fo.ViewField("created_at").month() <= 11)
autumn_view.tag_samples("autumn")
