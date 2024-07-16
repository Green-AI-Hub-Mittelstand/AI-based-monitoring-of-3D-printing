import json
import os

import requests
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials


def save_online_image_data_with_json_file(json_write_path, download_images=True):
    """
    Creates a custom vision client to a certain endpoint to retrieve the annotations of images with bounding boxes and
    labels. It saves the information to a json file. If download images is true, the images will be downloaded if they
    were not downloaded previously.
    :param json_write_path: path to the json file and images
    :param download_images: if True, images will be downloaded
    """
    api_key = ""
    endpoint = ""
    project_id = ""

    headers = {
        'Content-Type': 'application/json',
        'Training-key': api_key,
    }
    credentials = ApiKeyCredentials(in_headers={"Training-key": api_key})
    trainer = CustomVisionTrainingClient(endpoint, credentials)

    all_tagged_images = []
    batch_size = 256

    skip = 0

    while True:
        tagged_images = trainer.get_tagged_images(project_id, top=None, skip=skip, take=batch_size,
                                                  custom_headers=headers)

        all_tagged_images.extend(tagged_images)

        current_batch_count = len(tagged_images)
        skip += current_batch_count

        if current_batch_count == 0:
            break  # No more images to retrieve

    json_folder = os.path.dirname(json_write_path)
    if download_images:
        image_folder_path = os.path.join(json_folder, "Images")
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)
        i = 0
        for cvai_image in all_tagged_images:
            if str(cvai_image.id) + ".png" not in os.listdir(image_folder_path):
                download_image_data(cvai_image, image_folder_path)
                i += 1
        print('Added', i, 'new images.')

    json_data = [cvai_image.as_dict() for cvai_image in all_tagged_images]
    with open(json_write_path, "w") as json_file:
        json.dump(json_data, json_file)


def download_image_data(image, save_folder_path):
    """
    Downloads the image using its uri and saves it to the given folder path.
    :param image: image object
    :param save_folder_path: path to save the image to
    """
    if image is None:
        return

    try:
        image_name = str(image.id)
        image_write_path = os.path.join(save_folder_path, image_name + ".png")

        with open(image_write_path, "wb") as image_file:
            image_file.write(requests.get(image.original_image_uri).content)

    except:
        pass


if __name__ == '__main__':
    # Run to download the images
    json_path = "images/output.json"
    save_online_image_data_with_json_file(json_path, download_images=True)
