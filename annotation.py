import tensorflow as tf
print(tf.__version__)
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json


# Load the JSON data from the file
with open('laboro_tomato/_annotations/_test.json', 'r') as f:
    train_data = json.load(f)

categories = train_data["categories"]
annotations = train_data["annotations"]
images = train_data["images"]

# detail Classes
categories_id = [categories["id"] for categories in train_data["categories"]]
categories_name = [categories["name"] for categories in train_data["categories"]]
categories_supercategory = [categories["supercategory"] for categories in train_data["categories"]]
# detail images
images_id = [images["id"] for images in train_data["images"]]
images_file_name = [images["file_name"] for images in train_data["images"]]
images_coco_url = [images["coco_url"] for images in train_data["images"]]
images_height = [images["height"] for images in train_data["images"]]
images_width = [images["width"] for images in train_data["images"]]
images_date_captured = [images["date_captured"] for images in train_data["images"]]
images_flickr_url = [images["flickr_url"] for images in train_data["images"]]
# detail datasets
annotation_ids = [annotation["id"] for annotation in train_data["annotations"]]
annotation_category_id = [annotation["category_id"] for annotation in train_data["annotations"]]
annotation_image_id = [annotation["image_id"] for annotation in train_data["annotations"]]
annotation_segmentation = [annotation["segmentation"] for annotation in train_data["annotations"]]
annotation_iscrowd = [annotation["iscrowd"] for annotation in train_data["annotations"]]
annotation_bbox = [annotation["bbox"] for annotation in train_data["annotations"]]

# Create the table as before
rows = []
nums = 600 # change number for show detail and image at id
for annotation in annotations:
    if annotation["image_id"] == nums:
        id = annotation["image_id"]
        im_id = images_id.index(id)
        file_n = images_file_name[im_id]
        cat_id = annotation["category_id"]
        for category in categories:
            if category["id"] == cat_id:
                cat_name = category["name"]
        rows.append({"File Name": file_n, "Category ID": cat_id, "Category Name": cat_name, "Segmentation": annotation["segmentation"], "Bounding Box": annotation["bbox"], "Width": images_width[im_id], "Height": images_height[im_id]})
table = pd.DataFrame(rows)
# Display the table and images
image_show = Image.open('laboro_tomato/.train/{}'.format(file_n))
plt.imshow(image_show)
display(table)

class Image:
    def __init__(self, id, file_name, width, height):
        self.id = id
        self.file_name = file_name
        self.width = width
        self.height = height

class Category:
    def __init__(self, id, name, supercategory):
        self.id = id
        self.name = name
        self.supercategory = supercategory

class Annotation:
    def __init__(self, id, category_id, image_id, segmentation, bbox, iscrowd):
        self.id = id
        self.category_id = category_id
        self.image_id = image_id
        self.segmentation = segmentation
        self.bbox = bbox
        self.iscrowd = iscrowd


images = []
for image_data in train_data["images"]:
    id = image_data["id"]
    file_name = image_data["file_name"]
    width = image_data["width"]
    height = image_data["height"]
    image = Image(id, file_name, width, height)
    images.append(image)

categories = []
for category_data in train_data["categories"]:
    id = category_data["id"]
    name = category_data["name"]
    supercategory = category_data["supercategory"]
    category = Category(id, name, supercategory)
    categories.append(category)

annotations = []
for annotation_data in train_data["annotations"]:
    id = annotation_data["id"]
    category_id = annotation_data["category_id"]
    image_id = annotation_data["image_id"]
    segmentation = annotation_data["segmentation"]
    bbox = annotation_data["bbox"]
    iscrowd = annotation_data["iscrowd"]
    annotation = Annotation(id, category_id, image_id, segmentation, bbox, iscrowd)
    annotations.append(annotation)


train_data = {
    "images": [image.__dict__ for image in images],
    "categories": [category.__dict__ for category in categories],
    "annotations": [annotation.__dict__ for annotation in annotations],
}

with open("test_data.json", "w") as f:
    json.dump(train_data, f)
    print('succeses')
