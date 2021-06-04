import cv2
import numpy as np
import scipy.io as sio
from skimage import filters
import os
import argparse

shape = np.array((384,288),dtype=np.int)

def create_heatmap(landmarks):
    denum = 0.006366197723675813 # this const is value of point (0,0) in gauss kernel w mean
    heatmap = np.zeros((*shape,16), dtype=np.float)
    landmarks_bbox = landmarks[19] - landmarks[18]
    landmarks_transferred = landmarks[:18] - landmarks[18]
    new_landmarks = (landmarks_transferred[:,:2] / landmarks_bbox[:2] * shape[[1,0]]).astype(np.int)
    for i in range(16):
        landmark = new_landmarks[i]
        if landmark[0] >= 0 or landmark[1] >= 0:
            heatmap[landmark[1], landmark[0],i] = 1
            heatmap[:,:,i] = filters.gaussian(heatmap[:,:,i],sigma=5)/denum
    return heatmap


def crop_image(image,landmarks):
    bbox_left_corner = landmarks[18].astype(int)
    bbox_right_corner = landmarks[19].astype(int)
    result = image[bbox_left_corner[1]:bbox_right_corner[1],bbox_left_corner[0]:bbox_right_corner[0]]
    result = cv2.resize(result, tuple(shape[::-1]), interpolation=cv2.INTER_AREA)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile",type=str, default="train.txt",help="path to train db list")
    parser.add_argument("--ifolder",type=str,default="LV-MHP-v2",help="path to train dataset")
    parser.add_argument("--ofolder", type=str, default="output", help="output folder")
    parser.add_argument("--dbg", action="store_true",default=False)
    return parser.parse_args()


def find_all_landmarks(annot):
    landmark = []
    for name in annot.keys():
        if "person" in name:
            landmark.append(annot[name])
    return landmark


def main(args):
    with open(args.ifile,"r") as file:
        images = file.readlines()
    images = [image.strip() for image in images]
    out_image_path = os.path.join(args.ofolder, "cropped_image")
    out_heat_path = os.path.join(args.ofolder, "heatmap")
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_heat_path, exist_ok=True)
    if args.dbg:
        out_dbg_path = os.path.join(args.ofolder, "dbg")
        os.makedirs(out_dbg_path, exist_ok=True)
    with open(os.path.join(args.ofolder, "image.txt"),"w+") as all_image_list, \
            open(os.path.join(args.ofolder, "heatmaps.txt"),"w+") as all_heat_list:
        for image_name in images:
            image_path = os.path.join(args.ifolder,image_name + ".jpg")
            # image_path = os.path.join(args.ifolder,"train","images",image_name + ".jpg")
            # annot_path = os.path.join(args.ifolder,"train","pose_annos",image_name + ".mat")
            annot_path = os.path.join(args.ifolder,image_name + ".mat")
            annot = sio.loadmat(annot_path)
            landmarks = find_all_landmarks(annot)
            image = cv2.imread(image_path,-1)
            for idx,landmark_set in enumerate(landmarks):
                heatmap = create_heatmap(landmark_set)
                cropped_image = crop_image(image,landmark_set)
                full_path_img = os.path.join(out_image_path,image_name + f"_{idx}.jpg")
                full_path_heat = os.path.join(out_heat_path,image_name + f"_{idx}.npy")
                if args.dbg:
                    full_path_dbg = os.path.join(out_dbg_path, image_name + f"_{idx}.jpg")
                    cv2.imwrite(full_path_dbg,(np.clip(cropped_image/255 + np.sum(heatmap,axis=-1)[:,:,None],0,1)*255).astype(np.uint8))
                heatmap.tofile(full_path_heat)
                cv2.imwrite(full_path_img,cropped_image)
                all_image_list.write(full_path_img + "\n")
                all_heat_list.write(full_path_heat + "\n")
                print(f"{image_name} is done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
