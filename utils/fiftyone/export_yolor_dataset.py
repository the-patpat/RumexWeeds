#%% Get YOLOv5 tags (export the dataset in YOLO format) (single class)
import fiftyone as fo
import os

dataset = fo.load_dataset("RumexWeeds")

# Get class list
classes = dataset.default_classes


os.makedirs("/RumexWeeds-YOLOR/labels", exist_ok=True)
export_dir = "/RumexWeeds-YOLOR"

# Omit the _single to export the two-class dataset
label_field = "ground_truth_detections_single"  # for example

# The splits to export
splits = ["train", "val", "test"]

# All splits must use the same classes list
# Use ["rumex_obtusifolius", "rumex_crispus"] for the two-class dataset
classes = ["rumex"]

# The dataset or view to export
# We assume the dataset uses sample tags to encode the splits to export
# Export the splits
for split in splits:
    split_view = dataset.match_tags(split)
    split_view.export(
        labels_path=os.path.join(export_dir, "labels", split),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )

#%% make links
for split in ["train", "test", "val"]:
    with open(f'/RumexWeeds/dataset_splits/random_{split}.txt') as f:
        image_list = f.readlines()
        src_path = '/RumexWeeds/{0}'
        dst_path = '/RumexWeeds-YOLOR/images/{0}/{1}'
        os.makedirs(f'/RumexWeeds-YOLOR/images/{split}', exist_ok=True)
        for image_path in image_list:
            os.symlink(src_path.format(image_path.replace('\n', '')), dst_path.format(split, os.path.split(image_path.replace('\n', ''))[-1]) )
#%% yaml and names file
with open('/RumexWeeds-YOLOR/dataset.yaml', 'w') as f:
    f.write("""
names:
- rumex
nc: 1
train: /RumexWeeds-YOLOR/images/train/
val: /RumexWeeds-YOLOR/images/val/
test: /RumexWeeds-YOLOR/images/test/""")

with open('/RumexWeeds-YOLOR/dataset.names', 'w') as f:
    f.write("rumex")
