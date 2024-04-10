from ultralytics.utils.files import increment_path
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from ultralytics.utils import TQDM
import numpy as np
import json, os, tqdm, shutil, yaml, random
import pandas as pd
from ultralytics.data.converter import merge_multi_segment



def rescaleCategoryCOCOtoYOLO(annotationFile:dict):
    """
    This function Takes all the category IDs in the COCO format and rescale it 
    from the range [1, N] in [0, len(category)] for yolo formatting
    """
    assert type(annotationFile) == dict

    coco_annotation_mapping = {}
    category_counter = 0

    ## Update COCO annotations to bring all the category ids from 0 to 79
    # I take the max image ID from the original COCO dataset
    for category_struct in annotationFile["categories"]:
        if category_struct["id"] not in coco_annotation_mapping:
            coco_annotation_mapping[category_struct["id"]] = category_counter
            category_counter += 1
        
        category_struct["id"] = coco_annotation_mapping[category_struct["id"]]
    
    for annotation_struct in annotationFile["annotations"]:
        annotation_struct["category_id"] = coco_annotation_mapping[annotation_struct["category_id"]]

    return annotationFile



##### CODE COPIED AND MODIFIED FROM  -> from ultralytics.data.converter import convert_coco
def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        ```

    Output:
        Generates output files in the specified output directory.
    """

    # Create dataset directory
    save_dir = increment_path(save_dir, sep="_")  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    # coco80 = coco91_to_coco80_class()
    data = None
    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        data = rescaleCategoryCOCOtoYOLO(data)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                # cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                cls = ann["category_id"] # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
                    
    return data, save_dir



def loadAnnotations(all_annotationsPaths)-> dict:
    if type(all_annotationsPaths) == str:
        all_annotationsPaths = [all_annotationsPaths]
    
    all_annotations = {}
    for path in all_annotationsPaths: 
        path = os.path.normpath(path)
        dataset_name = path.split(os.sep)[-3]
    
        print(f"Loading {dataset_name} annotation file JSON from: {path} ")
        with open(path, "r" ) as fin:
            all_annotations[dataset_name] = json.load(fp=fin)
        print(f"Done.")
    
    return all_annotations



def mergeCocoAnnotations(annotations_dict):
        # Extract the first annotation
        
    merged_annotations = {
        "dataset_name" : "merged_dataset",
        "annotation_file": {
            # "info": [],
            # "licenses":[],
            # "categories":[],
            # "images":[],
            # "annotations":[],
        }
    }
    merged_annotations["annotation_file"] = annotations_dict[list(annotations_dict.keys())[0]].copy()
    
    # Iterate over each annotation starting from the second one
    for annotation_name, annotation in list(annotations_dict.items())[1:]:
        # Merge images and update IDs
        image_id_mapping = {}
        last_image_id = max(image['id'] for image in merged_annotations["annotation_file"]['images'])
        for image in annotation['images']:
            new_id = image['id'] + last_image_id
            image_id_mapping[image['id']] = new_id
            image['id'] = new_id
            merged_annotations["annotation_file"]['images'].append(image)
        
        # Merge categories and update IDs
        cat_id_mapping = {}
        last_cat_id = max(cat['id'] for cat in merged_annotations["annotation_file"]['categories'])
        for cat in annotation['categories']:
            if cat['name'] not in [c['name'] for c in merged_annotations["annotation_file"]['categories']]:
                last_cat_id += 1
                cat_id_mapping[cat['id']] = last_cat_id
                cat['id'] = last_cat_id
                merged_annotations["annotation_file"]['categories'].append(cat)
            else:
                existing_cat = next(c for c in merged_annotations["annotation_file"]['categories'] if c['name'] == cat['name'])
                cat_id_mapping[cat['id']] = existing_cat['id']
        
        # Merge annotations and update IDs
        last_ann_id = max(ann['id'] for ann in merged_annotations["annotation_file"]['annotations'])
        for ann in annotation['annotations']:
            ann['id'] += last_ann_id
            ann['image_id'] = image_id_mapping[ann['image_id']]
            ann['category_id'] = cat_id_mapping[ann['category_id']]
            merged_annotations["annotation_file"]['annotations'].append(ann)
    
    return merged_annotations

# def mergeCocoAnnotations(all_annotations:dict) -> dict:

#     """
#     Merge all other_annotationFiles in the first annotation in the dictionary:  first_annotation <- other_annotations
#     The merge is done in place, so it will modifies the first element in the dictionary.
#     To avoid this behaviour pass a copy in the dst_annotationFile
#     Return the modified dst_annotationFile
#     """

#     assert type(all_annotations) == dict
    

#     # print(f"Analyizing {dst_annotation['dataset_name']} annotation file")
#     # dst_annotationFile = dst_annotation['annotation_file']
#     # # Take the max of each ids            
#     # max_images_ids = max(dst_annotationFile["images"], key=lambda im: im["id"])["id"]
#     # max_annotation_ids = max(dst_annotationFile["annotations"], key=lambda an: an["id"])["id"]
#     # max_categories_ids = max(dst_annotationFile["categories"], key=lambda ca: ca["id"])["id"]
#     # max_licenses_ids = max(dst_annotationFile["licenses"], key=lambda ls: ls["id"])["id"]
#     # print("Done")
    
#     ## ==================================================================
    
#     max_images_ids = 0
#     max_categories_ids = 0
#     max_licenses_ids = 0
#     max_annotation_ids = 0
    
#     categories_names = []
#     image_names = []
#     license_names = []
    
#     conversion_licences_id_dict = {}
#     conversion_image_id_dict = {}
#     conversion_category_id_dict = {}
#     conversion_annotation_id_dict = {}
    
    
#     merged_annotation = {
#         "dataset_name" : "merged_dataset",
#         "annotation_file": {
#             "info": [],
#             "licenses":[],
#             "categories":[],
#             "images":[],
#             "annotations":[],
#         }
#     }
    
    
#     for dataset_name, annotationFile in all_annotations.items():
        
#         print(f"Serving {dataset_name} dataset:")


#         for licence_struct in tqdm.tqdm(annotationFile["licenses"], "Updating Licenses"):
            
#             if licence_struct["name"] not in license_names:
#                 license_names.append(licence_struct["name"])
                
#                 max_licenses_ids += 1

#                 conversion_licences_id_dict[licence_struct["id"]] = max_licenses_ids
#                 licence_struct["id"] = max_licenses_ids
                
#                 merged_annotation["annotation_file"]["licenses"].append(licence_struct.copy())
                


#         # Update image ids
#         for image_struct in tqdm.tqdm(annotationFile["images"], "Updating Images"):
            
#             if image_struct["file_name"] not in image_names:
#                 image_names.append(image_struct["file_name"])
                
#                 max_images_ids += 1

#                 conversion_image_id_dict[image_struct["id"]] = max_images_ids
#                 image_struct["id"] = max_images_ids
#                 image_struct["license"] = conversion_licences_id_dict[image_struct["license"]]
                
#                 merged_annotation["annotation_file"]["images"].append(image_struct.copy())
                
            

#         # Update Category ids
#         for category_struct in tqdm.tqdm(annotationFile["categories"], "Updating Categories"):

#             if category_struct["name"] not in categories_names:
#                 categories_names.append(category_struct["name"])
#                 max_categories_ids += 1

#                 conversion_category_id_dict[category_struct["id"]] = max_categories_ids
#                 category_struct["id"] = max_categories_ids
            
#                 merged_annotation["annotation_file"]["categories"].append(category_struct.copy())
            

#         # Update Annotation ids
#         for annotation_struct in tqdm.tqdm(annotationFile["annotations"], "Updating Annotations"):
#             max_annotation_ids += 1

#             conversion_annotation_id_dict[annotation_struct["id"]] = max_annotation_ids
#             annotation_struct["id"] = max_annotation_ids
#             annotation_struct["image_id"] = conversion_image_id_dict[annotation_struct["image_id"]]
#             annotation_struct["category_id"] = conversion_category_id_dict[annotation_struct["category_id"]]
            

#         # Update Coco-Dataset Annotation json
#         print("Merging COCO Datasets")
#         merged_annotation["annotation_file"]["licenses"].extend(annotationFile["licenses"])
#         merged_annotation["annotation_file"]["images"].extend(annotationFile["images"])
#         merged_annotation["annotation_file"]["categories"].extend(annotationFile["categories"])
#         merged_annotation["annotation_file"]["annotations"].extend(annotationFile["annotations"])
#         print("DONE\n")


#     return merged_annotation


def select_random_unique(group:pd.DataFrame, N, random_state = 42):
    np.random.seed(random_state)
    
    group = group.unique()
    
    ret = np.random.choice(group, size=min(N, len(group)), replace=False)
    
    # Reset random seed
    np.random.seed()
    
    return ret


def sample_images(group:pd.DataFrame, N:int, random_state:int) -> pd.DataFrame:
    # Determine the number of available images for this category
    num_images = len(group)
    # Adjust N if it's greater than the number of available images
    n_samples = min(N, num_images)
    # Sample the images randomly
    return group.sample(n=n_samples, random_state=random_state)



def CocoSubset(cocojson_annotationFile:dict, max_image_per_category:int, random_state:int = 42) -> dict:
    
    print(f"Choosing {max_image_per_category} images per category.")
    # Create a DataFrame from images data and annotations data
    subset_annotationFile = deepcopy(cocojson_annotationFile)
    image_df = pd.DataFrame.from_dict(subset_annotationFile["images"])
    annotation_df = pd.DataFrame.from_dict(subset_annotationFile["annotations"])
    

    # Sample randomically N images per category
    # selected_images = annotation_df.groupby(['category_id', 'image_id']).apply(sample_images, max_image_per_category, random_state)
    images_df = annotation_df.groupby('category_id')['image_id'].apply(select_random_unique, max_image_per_category, random_state)#.reset_index()

    filtered_images_ids = np.array([])
    for arr in images_df:
        filtered_images_ids = np.hstack((filtered_images_ids, arr))
        
    # Filter the images and the annotations
    # filtered_images = image_df[(image_df["id"].isin(selected_images["image_id"]))]
    # filtered_annotation = annotation_df[(annotation_df["image_id"].isin(selected_images["image_id"]))]
    
    filtered_images = image_df[(image_df["id"].isin(filtered_images_ids))]
    filtered_annotation = annotation_df[(annotation_df["image_id"].isin(filtered_images_ids))]

    print(f"We found {filtered_images.shape} Images and {filtered_annotation.shape} Annotations")

    filtered_images = filtered_images.to_dict(orient='records')
    filtered_annotation = filtered_annotation.to_dict(orient='records')

    # Update the Coco dataset Annotation
    print(f"We found {len(filtered_images)} Images and {len(filtered_annotation)} Annotations")
    subset_annotationFile["images"] = filtered_images
    subset_annotationFile["annotations"] = filtered_annotation

    return subset_annotationFile



def createMergedCocoDataset(newCocoAnnotation:dict, srcCocoDirs:list, dstCocoDir:str) -> None:

    allowed_image_ext = [".jpg", ".png"]
    allowed_dataset_split = ["train", "val", "test"]
    all_image_names = dict()

    dir_name = ""
    dataset_split = ""
    for srcDir in srcCocoDirs:
        path = os.path.normpath(srcDir)
        dir_name = os.path.dirname(path)
        dataset_split = path.split(os.sep)[-2]
        image_names = {im:dir_name for im in os.listdir(dir_name) if os.path.splitext(im)[1] in allowed_image_ext}
        all_image_names.update(image_names)
    
    print("Creating the new CocoDataset") # ===============================
    if dataset_split in allowed_dataset_split:
        dstCocoDir = os.path.join(dstCocoDir, dataset_split) 
    if os.path.exists(dstCocoDir):
        shutil.rmtree(dstCocoDir)
    os.makedirs(dstCocoDir)
    
    for image_struct in tqdm.tqdm(newCocoAnnotation["images"], desc="Coping new Images"):

        if image_struct["file_name"] in all_image_names.keys():
            shutil.copy(os.path.join(all_image_names[image_struct["file_name"]], image_struct["file_name"]), dstCocoDir)
        else:
            print(f"Image {image_struct['file_name']} not found")

    with open(os.path.join(dstCocoDir, "_annotations.coco.json"), "w") as fout:
        json.dump(newCocoAnnotation, fp=fout)
        
    print("Done\n") # ======================================================
    
    
    
def convertCOCOtoYOLOv8(cocoSrcDir, YoloDstDir, zipName=None, splits_perc = None):
    allowed_image_ext = [".jpg", ".png"]
    
   
    newCocoAnnotation, YoloDstDir = convert_coco(
        labels_dir=cocoSrcDir, 
        save_dir=YoloDstDir, 
        use_segments=True, 
        use_keypoints=False, 
        cls91to80=False
        )
    
    all_images = [im for im in os.listdir(cocoSrcDir) if os.path.splitext(im)[1] in allowed_image_ext]

    # Create structure for train test validation split
    if splits_perc is not None:
        train_images, val_images, test_images = split_data(all_images, splits_perc[0], splits_perc[1], splits_perc[2])
        for split_name, split_image in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            if os.path.exists(os.path.join(YoloDstDir, split_name, "images")) or os.path.exists(os.path.join(YoloDstDir, split_name, "labels")):
                shutil.rmtree(os.path.join(YoloDstDir, split_name, "images"))
                shutil.rmtree(os.path.join(YoloDstDir, split_name, "labels"))

            os.makedirs(os.path.join(YoloDstDir, split_name, "images"))
            os.makedirs(os.path.join(YoloDstDir, split_name, "labels"))
            
            for image in tqdm.tqdm(split_image, desc= f"Coping all images in the yolo dataset folder {split_name}"):
                src_image_dir = os.path.join(cocoSrcDir, image)
                dst_image_dir = os.path.join(YoloDstDir, split_name, "images")
                src_label_dir = os.path.join(YoloDstDir, "labels", "_annotations.coco", os.path.splitext(image)[0] + ".txt")
                dst_label_dir = os.path.join(YoloDstDir, split_name, "labels")
                try:
                    shutil.copy(src_label_dir, dst_label_dir)
                    shutil.copy(src_image_dir, dst_image_dir)
                except:
                    print(f"{os.path.splitext(image)[0]}.txt not found")

    else:
        for image in tqdm.tqdm(all_images, desc= f"Coping all images in the yolo dataset folder"):
            shutil.copy(os.path.join(cocoSrcDir, image), os.path.join(YoloDstDir, "images"))
        
    new_config_file = {category_struct["id"] : category_struct["name"] for category_struct in newCocoAnnotation["categories"]}
    
    if splits_perc is not None:   
        config_file = {
            "names": new_config_file,

            "test": "test/images",
            "val": "val/images",
            "train": "train/images",
        }
        
        shutil.rmtree(os.path.join(YoloDstDir, "images"))
        shutil.rmtree(os.path.join(YoloDstDir, "labels"))
    else:
        config_file = {
            "names": new_config_file,
            "train": "images",
        }
        

    with open(os.path.join(YoloDstDir, "data.yaml"), "w") as file: 
        yaml.dump(config_file, file)
    
    
    if zipName is not None:
        print(f"Creating the zip file: {zipName}")
        shutil.make_archive(zipName, 'zip', YoloDstDir)
        print(f"Done. Saved in {YoloDstDir}.")
    
        
        
def split_data(data, train_percent, val_percent, test_percent):
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate the sizes of each split
    num_samples = len(data)
    num_train = int(num_samples * train_percent)
    num_val = int(num_samples * val_percent)
    # num_test = int(num_samples * test_percent)

    # Split the data
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    return train_data, val_data, test_data



def modify_label_numbers(label_folders, data_yaml_path, label_to_modify, new_label):
    for folder in label_folders:
        print(folder)
        for filename in tqdm.tqdm(os.listdir(folder)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                with open(filepath, 'w') as f:
                    for line in lines:
                        class_index, *rest = line.strip().split(' ')
                        if int(class_index) == label_to_modify:
                            f.write(f"{new_label} {' '.join(rest)}\n")
                        else:
                            f.write(line)
    
    # Update data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'names' in data:
        names = data['names']
        if label_to_modify in names:
            names[new_label] = names.pop(label_to_modify)
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f)
        
        
def remove_labels_from_folders(label_folders, data_yaml_path, labels_to_remove):
    for folder in label_folders:
        print(folder)
        for filename in tqdm.tqdm(os.listdir(folder)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                with open(filepath, 'w') as f:
                    for line in lines:
                        class_index, *rest = line.strip().split(' ')
                        if int(class_index) not in labels_to_remove:
                            f.write(line)
    
    # Update data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'names' in data:
        names = data['names']
        for label_to_remove in labels_to_remove:
            if label_to_remove in names:
                del names[label_to_remove]
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f)
        
        
# def modify_label_numbers(label_folders, data_yaml_path, label_remap):
#     for folder in label_folders:
#         for filename in os.listdir(folder):
#             if filename.endswith(".txt"):
#                 filepath = os.path.join(folder, filename)
#                 with open(filepath, 'r') as f:
#                     lines = f.readlines()
#                 with open(filepath, 'w') as f:
#                     for line in lines:
#                         class_index, *rest = line.strip().split(' ')
#                         if int(class_index) in label_remap:
#                             new_label = label_remap[int(class_index)]
#                             f.write(f"{new_label} {' '.join(rest)}\n")
#                         else:
#                             f.write(line)
    
#     # Update data.yaml
#     with open(data_yaml_path, 'r') as f:
#         data = yaml.safe_load(f)
    
#     if 'names' in data:
#         names = data['names']
#         for current_label, new_label in label_remap.items():
#             if current_label in names:
#                 names[new_label] = names.pop(current_label)
    
#     with open(data_yaml_path, 'w') as f:
#         yaml.dump(data, f)


def mergeCOCOAndConvertToYOLO(merge_coco_paths:list[str],
                              creation_dst_coco: str, 
                              creation_dst_yolo:str,
                              max_images = None,
                              splits_perc:list[int] = [0.7, 0.2, 0.1],
                              zipName:str = None, 
                              random_state:int = 42) -> None:
    """
    This function merge the src dataset in the destination dataset.
    If max_images is defined the dataset will be clumped to that number of classes per category
    """
    all_annotations = loadAnnotations(merge_coco_paths)
    # for path in merge_coco_paths: 
    #     path = os.path.normpath(path)
    #     dataset_name = path.split(os.sep)[0]
    #     all_annotations[dataset_name] = loadAnnotations(path)
        
    # dst_annotations = all_annotations[0]
    # src_annotations = all_annotations[1:]
    
    print("Starting merge of annotations")
    merged_annotation = mergeCocoAnnotations(all_annotations)
    
    if max_images is not None:
        print("Starting subset selection")     
        MAX_IMAGES_PER_CATEGORY = min([len(annotation["images"]) for annotation in all_annotations.values()] + [max_images])
        merged_annotation["annotation_file"] = CocoSubset(merged_annotation["annotation_file"], max_image_per_category = MAX_IMAGES_PER_CATEGORY, random_state=random_state)
    
    print(f"Creating the new merged COCO dataset in folder: {creation_dst_coco}")
    createMergedCocoDataset(newCocoAnnotation=merged_annotation["annotation_file"], srcCocoDirs=merge_coco_paths, dstCocoDir=creation_dst_coco)
    
    print(f"Converting the new COCO dataset Yolo in folder: {creation_dst_yolo}")
    convertCOCOtoYOLOv8(creation_dst_coco+"\\train", creation_dst_yolo, zipName=zipName, splits_perc=splits_perc)
    