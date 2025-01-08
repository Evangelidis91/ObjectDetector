import fiftyone as fo
import fiftyone.zoo as foz
import os


class OpenImagesDatasetPreparation:
    def __init__(self, base_dir='dataset'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.classes = [
            'Shirt', 'Mobile phone', 'Laptop',
            'Car', 'Person', 'Man', 'Woman'
        ]

    def download_dataset(self, max_samples=5000):
        try:
            print("Starting dataset download...")

            datasets = {}
            splits = ["train", "validation", "test"]

            for split in splits:
                print(f"\nDownloading {split} split...")
                try:
                    dataset = foz.load_zoo_dataset(
                        "open-images-v7",
                        split=split,
                        label_types=["detections"],  # Only get detections
                        classes=self.classes,
                        max_samples=max_samples,
                        only_matching=True  # Only get samples that have annotations for our classes
                    )
                    print(f"Downloaded {len(dataset)} samples for {split} split")
                    datasets[split] = dataset
                except Exception as split_error:
                    print(f"Error downloading {split} split: {split_error}")
                    continue

            return datasets

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    def analyze_dataset(self, dataset):
        try:
            print("\nDataset Analysis:")
            print(f"Number of samples: {len(dataset)}")

            # Count detections for each class
            class_counts = {}
            samples_with_detections = 0
            samples_without_detections = 0

            for sample in dataset:
                try:
                    # Check if sample has detections
                    if hasattr(sample, 'detections') and sample.detections is not None:
                        samples_with_detections += 1
                        # Access the detections field directly
                        for detection in sample.detections.detections:
                            label = detection.label
                            class_counts[label] = class_counts.get(label, 0) + 1
                    else:
                        samples_without_detections += 1
                except AttributeError:
                    samples_without_detections += 1

            print("\nDetection Statistics:")
            print(f"Samples with detections: {samples_with_detections}")
            print(f"Samples without detections: {samples_without_detections}")

            if class_counts:
                print("\nObjects per class:")
                for label, count in class_counts.items():
                    print(f"  {label}: {count}")

            # Print some sample paths
            print("\nSample image paths:")
            for sample in list(dataset.take(5)):
                print(f"  {sample.filepath}")
                # Print detection info for this sample if available
                if hasattr(sample, 'detections') and sample.detections is not None:
                    print(f"    Detections: {len(sample.detections.detections)}")
                else:
                    print("    No detections")

        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            # Print more detailed error information
            import traceback
            print(traceback.format_exc())


def main():
    try:
        # Initialize dataset preparation
        dataset_prep = OpenImagesDatasetPreparation()

        # Download dataset
        print("Starting Open Images V7 download...")
        datasets = dataset_prep.download_dataset(max_samples=1000)

        if datasets:
            # Analyze each split
            for split_name, split_dataset in datasets.items():
                print(f"\nAnalyzing {split_name} split:")
                dataset_prep.analyze_dataset(split_dataset)
            print("\nDataset preparation completed successfully!")
        else:
            print("Dataset download failed.")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()