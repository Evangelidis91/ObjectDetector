import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import csv
import requests
from concurrent.futures import ThreadPoolExecutor
import json
from io import StringIO

from main import CLASSES


class OpenImagesDatasetPreparation:
    def __init__(self, base_dir='dataset'):
        """
        Initialize the dataset preparation class
        Args:
            base_dir (str): Base directory to store the dataset
        :param base_dir:
        """

        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.annotation_dir = os.path.join(base_dir, 'annotations')

        # Create necessary directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)

        # URLs for Open Images metadata
        self.class_descriptions_url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
        self.train_annotations_url = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv"
        self.validation_annotations_url = "https://storage.googleapis.com/openimages/v7/oidv7-validation-annotations-bbox.csv"
        self.test_annotations_url = "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-bbox.csv"

    def download_metadata(self):
        """
               Download and process the Open Images metadata files
               Returns:
                   dict: Mapping between class names and their OpenImages IDs
               """

        print("Downloading class descriptions...")
        response = requests.get(self.class_descriptions_url)
        class_desc = pd.read_csv(StringIO(response.text), header=None)

        # Create class name to ID mapping
        class_mapping = {}
        for _, row in class_desc.iterrows():
            class_id, class_name = row[0], row[1]
            if class_name in CLASSES:
                class_mapping[class_name] = class_id

        return class_mapping


    def download_annotations(self, class_mapping):
        """
                Download and filter annotations for our target classes
                Args:
                    class_mapping (dict): Mapping between class names and their OpenImages IDs
                """

        print("Downloading and processing annotations...")

        # Download annotations for each split
        splits = {
            'train': self.train_annotations_url,
            'validation': self.validation_annotations_url,
            'test': self.test_annotations_url
        }

        for split, url in splits.items():
            print(f"Processing {split} split...")

            # Download annotations
            response = requests.get(url)
            df = pd.read_csv(StringIO(response.text))

            # Filter annotations for our classes
            class_ids = list(class_mapping.values())
            filtered_df = df[df['LabelName'].isin(class_ids)]

            # Save filtered annotations
            output_path = os.path.join(self.annotation_dir, f'{split}_annotations.csv')
            filtered_df.to_csv(output_path, index = False)

    def download_image(self, image_id, image_url, split):
        """
               Download a single image
               Args:
                   image_id (str): Image identifier
                   image_url (str): URL to download the image
                   split (str): Dataset split (train/validation/test)
               """
        try:
            # Create split directory if it doesn't exist
            split_dir = os.path.join(self.image_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Download and save image
            response = requests.get(image_url)
            if response.status_code == 200:
                image_path = os.path.join(split_dir, f'{image_id}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error downloading image {image_id}: {str(e)}")



    def download_images(self, max_images_per_class=1000):
        """
        Download images for all splits
        Args:
            max_images_per_class (int): Maximum number of images to download per class
        """
        splits = ['train', 'validation', 'test']

        for split in splits:
            print(f"Downloading {split} images...")

            # Read annotations
            annotations_path = os.path.join(self.annotation_dir, f'{split}_annotations.csv')
            df = pd.read_csv(annotations_path)

            # Group by class and limit images per class
            for class_id in df['LabelName'].unique():
                class_df = df[df['LabelName'] == class_id].head(max_images_per_class)

                # Download images in parallel
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for _, row in class_df.iterrows():
                        executor.submit(
                            self.download_image,
                            row['ImageID'],
                            f"https://storage.googleapis.com/openimages/v7/{split}/{row['ImageID']}.jpg",
                            split
                        )

    def create_tf_records(self):
        """
        Convert the downloaded dataset into TFRecord format
        """
        splits = ['train', 'validation', 'test']

        for split in splits:
            print(f"Creating TFRecords for {split} split...")

            # Read annotations
            annotations_path = os.path.join(self.annotation_dir, f'{split}_annotations.csv')
            annotations_df = pd.read_csv(annotations_path)

            # Group annotations by image
            image_annotations = annotations_df.groupby('ImageID')

            # Create TFRecord file
            output_path = os.path.join(self.base_dir, f'{split}.tfrecord')
            with tf.io.TFRecordWriter(output_path) as writer:
                for image_id, group in tqdm(image_annotations):
                    # Read image
                    image_path = os.path.join(self.image_dir, split, f'{image_id}.jpg')
                    if not os.path.exists(image_path):
                        continue

                    with tf.io.gfile.GFile(image_path, 'rb') as fid:
                        encoded_image = fid.read()

                    # Create feature dictionary
                    feature = {
                        'image/encoded': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[encoded_image])),
                        'image/format': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b'jpg'])),
                        'image/object/bbox/xmin': tf.train.Feature(
                            float_list=tf.train.FloatList(value=group['XMin'].tolist())),
                        'image/object/bbox/xmax': tf.train.Feature(
                            float_list=tf.train.FloatList(value=group['XMax'].tolist())),
                        'image/object/bbox/ymin': tf.train.Feature(
                            float_list=tf.train.FloatList(value=group['YMin'].tolist())),
                        'image/object/bbox/ymax': tf.train.Feature(
                            float_list=tf.train.FloatList(value=group['YMax'].tolist())),
                        'image/object/class/label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=group['LabelName'].map(
                                {id: idx for idx, id in enumerate(group['LabelName'].unique())}
                            ).tolist()))
                    }

                    # Create Example and write to TFRecord
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())













