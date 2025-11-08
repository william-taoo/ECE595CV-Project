import json
from pycocotools.coco import COCO

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

process_dataset("coco/annotations/instances_train2017.json")