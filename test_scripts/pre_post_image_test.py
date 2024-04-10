from PIL import Image
import numpy as np
import albumentations


def preprocess_image(image):
    
    image = preprocessor(image=image)["image"]
    image = (image/255).astype(np.float32)

    return image


def postprocess_image(image):

    image = (image + 1.0) / 2.0  # -1,1 -> 0,1
    #image = image.transpose(2, 0, 1) # h, w, c -> c,h,w
    #image = image.numpy()
    image = (image * 255).astype(np.uint8)
    image = postprocessor(image=image)["image"]

    return image

if __name__ == "__main__":
    image_path = "Tests\\in_test_image.jpg"
    image = Image.open(image_path)

    if not image.mode == "RGB": image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)

    height, witdth, channels = image.shape
    new_size = 1000

    rescaler = albumentations.SmallestMaxSize(max_size = new_size)
    cropper = albumentations.CenterCrop(height=new_size, width=new_size)

    preprocessor = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size = new_size), 
        albumentations.CenterCrop(height=new_size, width=new_size)
        ])

    postprocessor = albumentations.Compose([
        albumentations.Resize(height= height, width=witdth)
    ])


    preprocessed_image = preprocess_image(image)
    postprocessed_image = postprocess_image(preprocessed_image)

    im = Image.fromarray(postprocessed_image)
    im.save("Tests\\out_test_image.jpg")