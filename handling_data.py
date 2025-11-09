import json
from pycocotools.coco import COCO
from random import sample
import shutil
import os

def process_dataset(path):
    coco = COCO(path)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    class_names = [cat['name'] for cat in cats]
    num_classes = len(class_names)
    num_images = len(coco.getImgIds())
    # print("Number of classes:", num_classes)
    # print("Classes:", class_names)
    # print("Number of images:", num_images)

    return num_classes, num_images

def get_annotation_info(path):
    with open(path) as f:
        data = json.load(f)

    print("Data keys: ", list(data.keys()))
    print("Image Example: ", {data["images"][0]})
    print("Annotation Example: ", {data["annotations"][0]})
    print("Categories Example: ", {data["categories"]})

def split_dataset(path, size=200, val_ratio=0.2):
    if size > 5000:
        raise ValueError("Dataset size is too large")
    
    with open(path) as f:
        data = json.load(f)

    image_ids = [img["id"] for img in data["images"]]
    # Randomly select `size` images
    selected_ids = sample(image_ids, size)
    split_point = int(size * (1 - val_ratio))
    train_ids = set(selected_ids[:split_point])
    val_ids = set(selected_ids[split_point:])

    def filter_coco(ids):
        imgs = [img for img in data["images"] if img["id"] in ids]
        anns = [ann for ann in data["annotations"] if ann["image_id"] in ids]
        return {
            "info": data["info"],
            "licenses": data["licenses"],
            "categories": data["categories"],
            "images": imgs,
            "annotations": anns
        }

    train_coco = filter_coco(train_ids)
    val_coco = filter_coco(val_ids)

    os.makedirs(f"coco/train2017_{size}", exist_ok=True)
    os.makedirs(f"coco/val2017_{size}", exist_ok=True)

    # Copy images into new folders
    for img in train_coco["images"]:
        shutil.copy(f"coco/train2017/{img['file_name']}",
                    f"coco/train2017_{size}/{img['file_name']}")
    for img in val_coco["images"]:
        shutil.copy(f"coco/train2017/{img['file_name']}",
                    f"coco/val2017_{size}/{img['file_name']}")

    # Save annotation files
    with open(f"coco/annotations/instances_train2017_{size}.json", "w") as f:
        json.dump(train_coco, f)
    with open(f"coco/annotations/instances_val2017_{size}.json", "w") as f:
        json.dump(val_coco, f)

    print(f"Created train/val splits with {split_point} train and {size - split_point} val images.")

process_dataset("coco/annotations/instances_train2017.json")