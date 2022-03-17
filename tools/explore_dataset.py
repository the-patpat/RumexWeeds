#%%
import fiftyone as fo
import os
import json
import glob
import torch 
from yolox.exp import get_exp
import sys
sys.path.insert(0, '/home/pat/YOLOX/tools/')
import demo


base_name = "RumexWeeds"
dataset_top_dir = "../data/"
data_path = "imgs/"
labels_path = "annotations.xml"

#%% Assemble the RumexWeeds dataset and make it persistent

#Read the train/test/val split (random)
d_split = {}
for x in ["test", "train", "val"]:
    with open(dataset_top_dir+f"dataset_splits/random_{x}.txt", 'r') as f:
        #Take abspath, easier to find later
        d_split[x] = [os.path.abspath(dataset_top_dir + loc_seq_pic.replace('\n', '')) for loc_seq_pic in f.readlines()]
        f.close()


if not fo.dataset_exists(base_name):
    # Create the dataset by creating the sub-datasets and appending them to a list
    # Iterate through every collection ({date}_{location}) and therein, iterate through every sequence
    datasets = []
    for collection in glob.glob(dataset_top_dir + '2021*'):
        for sub_dataset in glob.glob(collection + '/seq*'):
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
                    label_field="ground_truth",
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
    
    dataset.default_classes = ["rumex_obtusifolius", "rumex_crispus"]
    dataset.classes["ground_truth"] = ["rumex_obtusifolius", "rumex_crispus"]
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

#%% Load GPS data, train/test/val category 
for sample in dataset:
    if 'location' not in sample.field_names or sample['location'] is None:
        filepath, filename = os.path.split(sample.filepath)
        filename = os.path.splitext(filename)[0]
        with open(filepath + '/../gps.json', 'r') as f:
            gps_dict = json.load(f)
            gps_data = gps_dict[filename.replace('_rgb', '')]
            sample['location'] = fo.GeoLocation(point=[gps_data['longitude'], gps_data['latitude']])
            print(f"Added GPS info to sample {filename}: {sample['location']}\r", end="")
            sample.save()
            f.close()
    for split in d_split:
        if os.path.abspath(sample.filepath) in d_split[split]:
            sample.tags.append(split)
            sample.save()
    
#%% Launch the app 
session = fo.launch_app(dataset, auto=False)
#%% Create geodataframe
import geopandas
gdf = geopandas.GeoDataFrame()
from shapely.geometry import Point
d = {'filename' : [], 'geometry' : [], 'classes': []}
for sample in dataset:
    d['filename'].append(sample.filepath)
    d['geometry'].append(Point(sample['location']['point'][0],sample['location']['point'][1]))
    if sample.ground_truth_detections is not None:
        d['classes'].append(set([x['label'] for x in sample['ground_truth_detections'].detections if sample['ground_truth_detections']]))
    else:
        d['classes'].append(set())

#%%
import pandas as pd
import matplotlib.pyplot as plt
import folium

#Construct the geopandas df
gdf = geopandas.GeoDataFrame(pd.DataFrame(d))
m = folium.Map(location = [55.793307, 12.523117], tiles='OpenStreetMap', zoom_start = 9)
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry ]


def get_membership_from_classes(classes: set) -> str:
    """Encodes class membership of samples from the RumexWeeds dataset

    Arguments
    ------------
    classes: set of str
        A set of type str containing classes "rumex_obtusifolius" or "rumex_crispus" or both or none

    
    Output
    -------
    membership: str
        String that represents class membership
    """
    if len(classes) == 0:
            membership = "None"
    elif len(classes) == 1:
        if 'rumex_obtusifolius' in classes:
            membership = "Rumex obtusifolius"
        else:
            membership = "Rumex crispus"
    elif len(classes) == 2:
        membership = "Both"
    return membership 

def get_color_from_classes(classes: set) -> str:
    """Encodes class membership of samples from the RumexWeeds dataset

    Arguments
    ------------
    classes: set of str
        A set of type str containing classes "rumex_obtusifolius" or "rumex_crispus" or both or none

    
    Output
    -------
    color: str
        String that represents class membership through color:
        - green if no detection in sample
        - blue if rumex_obtusifolius only, regardless of cardinality
        - red if rumex_crispus only, regardless of cardinality
        - purple if both rumex_obtusifolius and rumex_crispus, regardless of cardinality

    """
    if len(classes) == 0:
            color = "green"
    elif len(classes) == 1:
        if 'rumex_obtusifolius' in classes:
            color = "blue"
        else:
            color = "red"
    elif len(classes) == 2:
        color = "purple"
    return color 

# Create Markers
for i, coordinates in enumerate(geo_df_list):
    #No detection is green, rumex o is blue, rumex c is red and both is purple
    color = get_color_from_classes(gdf.iloc[i, :].classes)
    html_str = f"Filename: {gdf.iloc[i, :].filename} <br>"# + \
                #f"<img src={gdf.iloc[i, :].filename.replace('/mnt/d/OneDrive - Danmarks Tekniske Universitet/Thesis/Experiments/RumexWeeds/data/', 'file://D:/RumexWeeds/')} width=100 alt=\"none\"/>"
    #print(f"Color: {color}")
    m.add_child(folium.CircleMarker(location = coordinates, popup=html_str, color=color, radius=3))#, icon = folium.Icon(color = color)))

#Save the map html to disk
with open('samples.html', 'w') as f:
    f.write(str(m._repr_html_()))

#%% Create interactive geo-spatial plot
from fiftyone import ViewField as F
plot = fo.location_scatterplot(
    samples=dataset,
    labels=[get_membership_from_classes(set(x)) if x is not None else get_membership_from_classes(set()) for x in dataset.values('ground_truth_detections.detections.label')],
    sizes=[len(x)+1 if x is not None else 1 for x in dataset.values('ground_truth_detections.detections')]
)
with open('.token', 'r') as f: token = f.read()
plot.update_layout(mapbox_style="satellite", mapbox_accesstoken=token)
plot.show()
session.plots.attach(plot)
plot.save('samples_satellite.html')
session.show()
#%%
bb_hist = fo.NumericalHistogram(F('ground_truth_detections.detections[]').apply(F('bounding_box')[0]), init_view = dataset)
#%% Area histogram
bb_area_hist = fo.NumericalHistogram(F('ground_truth_detections.detections[]').apply(F('bounding_box')[2] * F('bounding_box')[3]), init_view=dataset)
session.plots.attach(bb_area_hist)
bb_area_hist.show()

#%%BB dimension clustering
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

#Get bb dimensions
#bounding_box: [<top-left-x>, <top-left-y>, <width>, <height>] (relative to image dimensions, [0, 1])

bb_width_height = np.asarray(dataset.values(F('ground_truth_detections.detections[]').apply(F('bounding_box'))[2:4]))
print(bb_width_height.shape)
km = KMeans(n_clusters=5, random_state=0)

km.fit(bb_width_height)
fig, ax = plt.subplots(1)
ax.set_ylim(0, 1200)
ax.set_xlim(0, 1920)
ax.invert_yaxis()
#ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1])

#Get bb top left/right
bb_top_left_corner = np.asarray(dataset.values(F('ground_truth_detections.detections[]').apply(F('bounding_box'))[0:2]))
print(bb_width_height.shape)
km_corner = KMeans(n_clusters=5, random_state=0)

km_corner.fit(bb_top_left_corner)
ax.scatter(km_corner.cluster_centers_[:, 0]*1920, km_corner.cluster_centers_[:, 1]*1200)

for dimension, anchor in zip(km.cluster_centers_, km_corner.cluster_centers_):
    ax.add_collection(PatchCollection([Rectangle((anchor[0]*1920, anchor[1]*1200), dimension[0]*1920, dimension[1]*1200)], alpha=0.2))
ax.set_title("Anchor Boxes on RumexWeeds")

#%% Load model YOLOX-darknet53

state_dict = torch.load('/home/pat/YOLOX/YOLOX_outputs/yolox_DarkNet53_rumexweeds/best_ckpt.pth', map_location='cpu')
exp = get_exp('/home/pat/YOLOX/exps/my_exps/yolox_DarkNet53_rumexweeds.py', 'DN53_rumexweeds')
model = exp.get_model()
model.cuda()
model.load_state_dict(state_dict["model"])
#Turns off some normalization and dropout layers, sets training variables to false
model.eval()
#%% 
#from https://voxel51.com/docs/fiftyone/recipes/adding_detections.html?highlight=model
from PIL import Image
from torchvision.transforms import functional as func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Add predictions to samples
def add_preds_yolox(predictions_view, device, predictor, field_name="predictions"):
    with fo.ProgressBar() as pb:
        for sample in pb(predictions_view):
            # Load image
            if sample.has_field(field_name):
                if sample[field_name] is not None:
                    continue
            image = Image.open(sample.filepath)
            image = func.to_tensor(image).to(device)
            image = image.float()
            image = image.half()
            c, h, w = image.shape

            # Perform inference
            
            preds, img_info = predictor.inference(sample.filepath)
            preds = preds[0]
            boxes = preds[:, 0:4].cpu().detach().numpy()

            # preprocessing: resize
            boxes /= img_info["ratio"]

            labels = preds [:, 6].cpu().detach().numpy().astype(int)
            scores = (preds[:, 4] * preds[:, 5]).cpu().detach().numpy()
    #preds = model(image.half())[0]

            # Convert detections to FiftyOne format
            detections = []
            for label, score, box in zip(labels, scores, boxes):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

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
#Take samples
predictions_view = dataset.take(100, seed=51)

# Get class list
classes = dataset.default_classes

#YOLOX specific predictor
#%%YOLO DN 53
predictor = demo.Predictor(model, exp, exp.classes_to_consider, None, None, 'gpu', False, False) 
add_preds_yolox(dataset, device, predictor, field_name=f"predictions_{exp.exp_name}")
#%%  Load yolox s outputs
state_dict = torch.load('/home/pat/YOLOX/YOLOX_outputs/yolox_s_rumexweeds/best_ckpt_single.pth', map_location='cpu')
exp = get_exp('/home/pat/YOLOX/exps/my_exps/yolox_s_rumexweeds.py', 'yolox_s_rumexweeds')
model = exp.get_model()
model.cuda()
model.load_state_dict(state_dict["model"])
#Turns off some normalization and dropout layers, sets training variables to false
model.eval()
#YOLOX specific predictor
predictor = demo.Predictor(model, exp, exp.classes_to_consider, None, None, 'gpu', False, False) 
add_preds_yolox(dataset, device, predictor, field_name=f"predictions_{exp.exp_name}")
#%%  Load yolox s outputs
state_dict = torch.load('/home/pat/YOLOX/YOLOX_outputs/yolox_l_rumexweeds/best_ckpt.pth', map_location='cpu')
exp = get_exp('/home/pat/YOLOX/exps/my_exps/yolox_l_rumexweeds.py', 'yolox_l_rumexweeds')
model = exp.get_model()
model.cuda()
model.load_state_dict(state_dict["model"])
#Turns off some normalization and dropout layers, sets training variables to false
model.eval()
#YOLOX specific predictor
predictor = demo.Predictor(model, exp, exp.classes_to_consider, None, None, 'gpu', False, False) 
add_preds_yolox(dataset, device, predictor, field_name=f"predictions_{exp.exp_name}")

#%% Get YOLOv5 tags
from fiftyone.utils.yolo import YOLOv5DatasetExporter

export_dir = "~/onedrive/Thesis/Experiments/RumexWeeds-YOLOv5/labels/"

label_field = "ground_truth_detections"  # for example

# The splits to export
splits = ["train", "val", "test"]

# All splits must use the same classes list
classes = ["rumex_obtusifolius"]

# The dataset or view to export
# We assume the dataset uses sample tags to encode the splits to export

# Export the splits
for split in splits:
    split_view = dataset.match_tags(split)
    split_view.export(
        labels_path=export_dir+split,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )
