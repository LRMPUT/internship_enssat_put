import os
import pandas as pd
from PIL import Image, ImageEnhance
import random

# Define transformations
def augment_image(image):
    transform_type = random.choice(["rotate", "flip", "color_jitter"])

    if transform_type == "rotate":
        angle = random.randint(-10, 10)  #rotate between -10 and 10 degrees
        image = image.rotate(angle)

    elif transform_type == "flip":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip

    elif transform_type == "color_jitter":
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.8, 1.2)  # color variation
        image = enhancer.enhance(factor)

    return image

csv_file = "./data.csv"
root_dir = "." 
augmented_dir = "images_augmented"  

os.makedirs(augmented_dir, exist_ok=True)
df = pd.read_csv(csv_file)
augmented_data = []
for index, row in df.iterrows():
    img_path = os.path.join(root_dir, row[0])
    label = row[1]

    #modifiy the image
    image = Image.open(img_path).convert("RGB")
    augmented_image = augment_image(image)

    # save the new image
    new_img_name = f"aug_{index}.jpg"
    new_img_path = os.path.join(augmented_dir, new_img_name)
    augmented_image.save(new_img_path)

    augmented_data.append([new_img_name, label])

# convert to DataFrame and save new CSV
augmented_df = pd.DataFrame(augmented_data, columns=["filename", "label"])
new_csv_file = "dataset_augmented.csv"
augmented_df.to_csv(new_csv_file, index=False)

print(f"Augmented dataset saved to {new_csv_file} with {len(augmented_data)} new images.")
