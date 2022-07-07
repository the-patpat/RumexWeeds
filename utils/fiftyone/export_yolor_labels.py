#%% Get YOLOv5 tags (export the dataset in YOLO format) (single class)
import fiftyone as fo
import os

dataset = fo.load_dataset("RumexWeeds")

# Get class list
classes = dataset.default_classes


os.makedirs("/RumexWeeds-YOLOR")
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
        labels_path=export_dir+split,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )
