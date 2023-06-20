import os
import xml.etree.ElementTree as ET
from PIL import Image

# Define a dictionary to map object classes to numerical labels
class_labels = {"apple": 0, "banana": 1, "orange": 2}

for fol in ["train", "test"]:
    names = os.listdir(fol)
    images_dir = fol + "/images"
    labels_dir = fol + "/labels"
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for name in names:
        file_tuple = os.path.splitext(name)
        if file_tuple[1].lower() in [".jpg", ".jpeg", ".png"]:
            image_path = os.path.join(fol, name)
            with open(image_path, "rb") as read, open(os.path.join(images_dir, name), "wb") as write:
                write.write(read.read())
        elif file_tuple[1] == ".xml":
            tree = ET.parse(os.path.join(fol, name))
            root = tree.getroot()

            img_width_elem = root.find('size').find('width')
            img_height_elem = root.find('size').find('height')

            img_width = int(img_width_elem.text)
            img_height = int(img_height_elem.text)

            if img_width == 0 or img_height == 0:
                image = Image.open(os.path.join(fol, file_tuple[0] + ".jpg"))
                img_width, img_height = image.size
                image.close()

            for obj in root.findall('object'):
                object_class = obj.find('name').text.lower()

                if object_class not in class_labels:
                    continue

                label = class_labels[object_class]

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                x_center = (xmin + xmax) / (2 * img_width)
                y_center = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                label_line = f"{label} {x_center} {y_center} {width} {height}"

                label_filename = file_tuple[0] + ".txt"
                label_path = os.path.join(labels_dir, label_filename)
                with open(label_path, "a") as label_file:
                    label_file.write(label_line + "\n")
