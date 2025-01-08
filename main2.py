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

# Define our target classes
CLASSES = [
    'Door', 'Shoe', 'Shirt', 'Keyboard', 'Mobile phone', 'Laptop',
    'Computer', 'Food', 'Fruit', 'Vegetable', 'Car', 'Person',
    'Man', 'Woman'
]


class OpenImagesDatasetPreparation:
    def __init__(self, base_dir='dataset'):
        """
        Initialize the dataset preparation class
        Args:
            base_dir (str): Base directory to store the dataset
        """
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, 'images')
        self.annotation_dir = os.path.join(base_dir, 'annotations')

        # Create necessary directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)

        # URLs for Open Images metadata
        self.class_descriptions_url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
        self.train_annotations_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
        self.validation_annotations_url = "https://storage.googleapis.com/openimages/v6/validation-annotations-bbox.csv"
        self.test_annotations_url = "https://storage.googleapis.com/openimages/v6/test-annotations-bbox.csv"

    def download_metadata(self):
        """
        Download and process the Open Images metadata files
        Returns:
            dict: Mapping between class names and their OpenImages IDs
        """
        try:
            print("Downloading class descriptions...")
            response = requests.get(self.class_descriptions_url)
            response.raise_for_status()  # Raise an error for bad responses

            # Read CSV with explicit column names
            class_desc = pd.read_csv(
                StringIO(response.text),
                names=['LabelID', 'ClassName'],  # Explicitly name the columns
                header=None
            )

            # Create class name to ID mapping
            class_mapping = {}
            for _, row in class_desc.iterrows():
                if row['ClassName'] in CLASSES:
                    class_mapping[row['ClassName']] = row['LabelID']

            print(f"Found {len(class_mapping)} classes out of {len(CLASSES)} requested classes")
            missing_classes = set(CLASSES) - set(class_mapping.keys())
            if missing_classes:
                print(f"Warning: Could not find the following classes: {missing_classes}")

            return class_mapping

        except requests.RequestException as e:
            print(f"Error downloading class descriptions: {e}")
            return {}
        except pd.errors.EmptyDataError:
            print("Error: Downloaded class descriptions file is empty")
            return {}
        except Exception as e:
            print(f"Unexpected error while processing class descriptions: {e}")
            return {}

    def download_annotations(self, class_mapping):
        """
        Download and filter annotations for our target classes
        Args:
            class_mapping (dict): Mapping between class names and their OpenImages IDs
        """
        if not class_mapping:
            print("Error: No class mapping available. Skipping annotation download.")
            return

        print("Downloading and processing annotations...")

        # Download annotations for each split
        splits = {
            'train': self.train_annotations_url,
            'validation': self.validation_annotations_url,
            'test': self.test_annotations_url
        }

        for split, url in splits.items():
            try:
                print(f"Processing {split} split...")

                # Download annotations
                response = requests.get(url)
                response.raise_for_status()

                # Read CSV with explicit column names
                df = pd.read_csv(StringIO(response.text))

                # Print column names for debugging
                print(f"Available columns in {split} annotations: {df.columns.tolist()}")

                # Verify required columns exist
                required_columns = ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Error: Missing required columns in {split} annotations: {missing_columns}")
                    continue

                # Filter annotations for our classes
                class_ids = list(class_mapping.values())
                filtered_df = df[df['LabelName'].isin(class_ids)]

                # Print statistics
                print(f"Found {len(filtered_df)} annotations for {split} split")

                # Save filtered annotations
                output_path = os.path.join(self.annotation_dir, f'{split}_annotations.csv')
                filtered_df.to_csv(output_path, index=False)
                print(f"Saved filtered annotations to {output_path}")

            except requests.RequestException as e:
                print(f"Error downloading {split} annotations: {e}")
            except pd.errors.EmptyDataError:
                print(f"Error: Downloaded {split} annotations file is empty")
            except Exception as e:
                print(f"Unexpected error while processing {split} annotations: {e}")



    def download_image(self, image_id, split):
        """
        Download a single image using CVDF mirror
        Args:
            image_id (str): Image identifier
            split (str): Dataset split (train/validation/test)
        """
        try:
            # Create split directory if it doesn't exist
            split_dir = os.path.join(self.image_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Check if image already exists
            image_path = os.path.join(split_dir, f'{image_id}.jpg')
            if os.path.exists(image_path):
                return

            # Construct the URL using CVDF mirror
            url = f"https://cvdf.githubusercontent.com/open-images/v7/{split}/{image_id}.jpg"

            # Download and save image
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {image_id}")

        except requests.RequestException as e:
            print(f"Error downloading image {image_id}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error while downloading image {image_id}: {str(e)}")

    def download_images(self, max_images_per_class=1000):
        """
        Download images for all splits
        Args:
            max_images_per_class (int): Maximum number of images to download per class
        """
        splits = ['train', 'validation', 'test']

        for split in splits:
            try:
                print(f"Downloading {split} images...")

                # Read annotations
                annotations_path = os.path.join(self.annotation_dir, f'{split}_annotations.csv')
                if not os.path.exists(annotations_path):
                    print(f"Error: Annotations file not found for {split} split")
                    continue

                df = pd.read_csv(annotations_path)

                # Group by class and limit images per class
                for class_id in df['LabelName'].unique():
                    class_df = df[df['LabelName'] == class_id].head(max_images_per_class)

                    print(f"Downloading {len(class_df)} images for class {class_id} in {split} split")

                    # Download images in parallel
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = []
                        for _, row in class_df.iterrows():
                            futures.append(
                                executor.submit(
                                    self.download_image,
                                    row['ImageID'],
                                    split
                                )
                            )

                        # Wait for all downloads to complete
                        for future in tqdm(futures, desc=f"Downloading {split} images"):
                            future.result()

            except Exception as e:
                print(f"Error processing {split} split: {str(e)}")


def main():
    """
    Main function to run the dataset preparation pipeline
    """
    try:
        # Initialize dataset preparation
        dataset_prep = OpenImagesDatasetPreparation()

        # Download and process metadata
        print("Step 1: Downloading metadata...")
        class_mapping = dataset_prep.download_metadata()
        if not class_mapping:
            raise Exception("Failed to create class mapping")

        # Save class mapping for reference
        mapping_path = os.path.join(dataset_prep.base_dir, 'class_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"Saved class mapping to {mapping_path}")

        # Download and process annotations
        print("\nStep 2: Downloading annotations...")
        dataset_prep.download_annotations(class_mapping)

        # Download images
        print("\nStep 3: Downloading images...")
        dataset_prep.download_images(max_images_per_class=1000)

        print("\nDataset preparation completed successfully!")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()