
import os
import cv2
import torch
import random
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ----------------------------
# CONFIG
# ----------------------------

DATASETS = [
    "/content/drive/MyDrive/PlantClinic_AI/datasets/plantvillage",
    "/content/drive/MyDrive/PlantClinic_AI/datasets/plantdoc"
]

OUTPUT_DATASET = "/content/drive/MyDrive/PlantClinic_AI/leaf_detection_dataset"

SAM_CHECKPOINT = "/content/drive/MyDrive/PlantClinic_AI/models/sam_vit_b_01ec64.pth"

MAX_IMAGES = 2000


# ----------------------------
# Create dataset folders
# ----------------------------

images_out = os.path.join(OUTPUT_DATASET,"images")
labels_out = os.path.join(OUTPUT_DATASET,"labels")

os.makedirs(images_out,exist_ok=True)
os.makedirs(labels_out,exist_ok=True)


# ----------------------------
# Collect image paths
# ----------------------------

image_paths = []

for dataset in DATASETS:

    for root,dirs,files in os.walk(dataset):

        for f in files:

            if f.lower().endswith((".jpg",".jpeg",".png")):
                image_paths.append(os.path.join(root,f))


print("Total leaf images:",len(image_paths))

sample_images = random.sample(image_paths,MAX_IMAGES)

print("Processing images:",len(sample_images))


# ----------------------------
# Load SAM model
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.9,
    min_mask_region_area=2000
)

print("SAM loaded on:",device)


# ----------------------------
# IoU function
# ----------------------------

def compute_iou(box1, box2):

    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    xa = max(x1,x2)
    ya = max(y1,y2)
    xb = min(x1+w1,x2+w2)
    yb = min(y1+h1,y2+h2)

    inter = max(0,xb-xa) * max(0,yb-ya)
    union = w1*h1 + w2*h2 - inter

    if union == 0:
        return 0

    return inter/union


# ----------------------------
# Main generator
# ----------------------------

for i,img_path in enumerate(sample_images):

    if i % 50 == 0:
        print("Processed:",i)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img,(640,640))

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img_rgb)

    boxes = []

    for m in masks:

        area = m["area"]

        if area < 2000 or area > 80000:
            continue

        mask = m["segmentation"].astype("uint8")

        x,y,w,h = cv2.boundingRect(mask)

        if min(w,h) < 40:
            continue

        aspect = w/h

        if aspect < 0.3 or aspect > 4:
            continue

        leaf = img_rgb[y:y+h,x:x+w]

        hsv = cv2.cvtColor(leaf,cv2.COLOR_RGB2HSV)

        veg_mask = cv2.inRange(hsv,(20,30,30),(120,255,255))

        veg_ratio = np.sum(veg_mask>0)/(w*h)

        if veg_ratio < 0.25:
            continue

        gray = cv2.cvtColor(leaf,cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray,100,200)

        edge_density = np.sum(edges>0)/(w*h)

        if edge_density < 0.01:
            continue

        boxes.append((x,y,w,h))


    # merge overlapping boxes
    merged = []

    for box in boxes:

        merged_flag = False

        for j,mb in enumerate(merged):

            if compute_iou(box,mb) > 0.4:

                x1 = min(box[0],mb[0])
                y1 = min(box[1],mb[1])

                x2 = max(box[0]+box[2],mb[0]+mb[2])
                y2 = max(box[1]+box[3],mb[1]+mb[3])

                merged[j] = (x1,y1,x2-x1,y2-y1)

                merged_flag = True
                break

        if not merged_flag:
            merged.append(box)


    if len(merged) == 0:
        continue


    H,W = img_rgb.shape[:2]

    label_lines = []

    for x,y,w,h in merged:

        xc = (x + w/2)/W
        yc = (y + h/2)/H
        wn = w/W
        hn = h/H

        label_lines.append(f"0 {xc} {yc} {wn} {hn}")


    img_name = os.path.basename(img_path)

    cv2.imwrite(os.path.join(images_out,img_name),img)

    label_name = img_name.replace(".jpg",".txt")

    with open(os.path.join(labels_out,label_name),"w") as f:
        f.write("\n".join(label_lines))


print("Pseudo-label generation finished")
