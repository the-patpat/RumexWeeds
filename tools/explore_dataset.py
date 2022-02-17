#%%
import fiftyone as fo
import os
import json
import glob

base_name = "RumexWeeds"
dataset_top_dir = "../data/20210806_hegnstrup/"
data_path = "imgs/"
labels_path = "annotations.xml"

# Create the dataset
datasets = []
for sub_dataset in glob.glob(dataset_top_dir+'seq*'):
    collection_name, seq_name = os.path.split(sub_dataset)
    collection_name = os.path.split(collection_name)[-1]
    name = f"{base_name}_{collection_name}_{seq_name}"
    dataset_dir = dataset_top_dir + seq_name
    if not fo.dataset_exists(name):
        datasets.append(fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.CVATImageDataset,
            name=name,
            data_path=data_path,
            labels_path=labels_path
        ))
    else:
        datasets.append(fo.load_dataset(name))

# View summary info about the dataset
print(datasets)
#%%
dataset = fo.Dataset()
for sequence in datasets:
    dataset.add_samples(iter(sequence))
view = dataset.view()
print(view)

#%% Load GPS data
for sample in dataset:
    filepath, filename = os.path.split(sample.filepath)
    filename = os.path.splitext(filename)[0]
    with open(filepath + '/../gps.json', 'r') as f:
        gps_dict = json.load(f)
        gps_data = gps_dict[filename.replace('_rgb', '')]
        f.close()
        sample['location'] = fo.GeoLocation(point=[gps_data['longitude'], gps_data['latitude']])

#%%
gps_data['longitude']
# %%

