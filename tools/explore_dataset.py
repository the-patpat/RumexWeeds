#%%
import fiftyone as fo
import os
import json
import glob

base_name = "RumexWeeds"
dataset_top_dir = "../data/"
data_path = "imgs/"
labels_path = "annotations.xml"

#%% Assemble the RumexWeeds dataset and make it persistent
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
                    labels_path=labels_path
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

#%% Load GPS data
for sample in dataset:
    if sample['location'] is None:
        filepath, filename = os.path.split(sample.filepath)
        filename = os.path.splitext(filename)[0]
        with open(filepath + '/../gps.json', 'r') as f:
            gps_dict = json.load(f)
            gps_data = gps_dict[filename.replace('_rgb', '')]
            sample['location'] = fo.GeoLocation(point=[gps_data['longitude'], gps_data['latitude']])
            print(f"Added GPS info to sample {filename}: {sample['location']}\r", end="")
            sample.save()
            f.close()
#%% Launch the app 
session = fo.launch_app(dataset, auto=False)
#%%
import geopandas
gdf = geopandas.GeoDataFrame()
#%%
from shapely.geometry import Point
d = {'filename' : [], 'geometry' : [], 'classes': []}
for sample in dataset:
    d['filename'].append(sample.filepath)
    d['geometry'].append(Point(sample['location']['point'][0],sample['location']['point'][1]))
    if sample.detections is not None:
        d['classes'].append(set([x['label'] for x in sample['detections'].detections if sample['detections']]))
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

# Create Markers
for i, coordinates in enumerate(geo_df_list):
    if i % 10 == 0:
        #No detection is green, rumex o is blue, rumex c is red and both is purple
        if len(gdf.iloc[i, :].classes) == 0:
            color = "green"
        elif len(gdf.iloc[i, :].classes) == 1:
            if 'rumex_obtusifolius' in gdf.iloc[i, :].classes:
                color = "blue"
            else:
                color = "red"
        elif len(gdf.iloc[i, :].classes) == 2:
            color = "purple"
        html_str = f"Filename: {gdf.iloc[i, :].filename} <br>"# + \
                   #f"<img src={gdf.iloc[i, :].filename.replace('/mnt/d/OneDrive - Danmarks Tekniske Universitet/Thesis/Experiments/RumexWeeds/data/', 'file://D:/RumexWeeds/')} width=100 alt=\"none\"/>"
        #print(f"Color: {color}")
        m.add_child(folium.CircleMarker(location = coordinates, popup=html_str, color=color, radius=3))#, icon = folium.Icon(color = color)))

#Save the map html to disk
with open('samples.html', 'w') as f:
    f.write(str(m._repr_html_()))
#%%
