import os, shutil, tqdm

IMAGE_PER_CHUNKS = 200

FOLDER_SEEED_NAME = "dataset_chunk"
SRC_DIR = "GigaDataset.v4i.yolov8"
DST_DIR = "DatasetChunks"

def extraxtAllFile(path:str) -> list:
    norm_path = os.path.normpath(path)

    return sorted([p for p in os.listdir(norm_path) if os.path.isfile(os.path.join(norm_path, p))])

images_path = f"{SRC_DIR}\\images"
labels_path = f"{SRC_DIR}\\labels"

all_images_path = extraxtAllFile(images_path)
all_labels_path = extraxtAllFile(labels_path)

images_chunks = [all_images_path[x:x+IMAGE_PER_CHUNKS] for x in range(0, len(all_images_path), IMAGE_PER_CHUNKS)]
labels_chunks = [all_labels_path[x:x+IMAGE_PER_CHUNKS] for x in range(0, len(all_labels_path), IMAGE_PER_CHUNKS)]

assert sum([len(c) for c in images_chunks]) == len(all_images_path) and len(all_images_path) == len(all_labels_path)

for i, (chunks_images_paths, chunks_labels_paths) in enumerate(zip(images_chunks, labels_chunks)):

    print(f"Creating Chunk Number {i}")
    save_dir = f"{DST_DIR}/{FOLDER_SEEED_NAME}_{i}"

    os.makedirs(f"{save_dir}/images")
    os.makedirs(f"{save_dir}/labels")

    for image_path, label_path in tqdm.tqdm(zip(chunks_images_paths, chunks_labels_paths)):

        image_filename, image_file_extension = os.path.splitext(image_path)
        label_filename, label_file_extension = os.path.splitext(label_path)

        assert image_filename == label_filename and image_file_extension == ".jpg" and label_file_extension == ".txt"

        shutil.copy(f"{SRC_DIR}/images/{image_path}", f"{save_dir}/images/{image_path}")
        shutil.copy(f"{SRC_DIR}/labels/{label_path}", f"{save_dir}/labels/{label_path}")

    print(f"Zipping file {FOLDER_SEEED_NAME}_{i}")
    shutil.make_archive(save_dir, 'zip', save_dir)

    shutil.rmtree(save_dir)