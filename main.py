from ImagePreprocessing import ImagePreprocessor
from OpenImagesDatasetPreparation import OpenImagesDatasetPreparation

PATH_NAME = "/Users/konstantinosevangelidis/fiftyone/open-images-v7"
# Example single image processing
image_path = PATH_NAME + "/train/data/037773834941655c.jpg"

image_paths = [
    PATH_NAME + "/train/data/034473474965116d.jpg",
    PATH_NAME + "/train/data/0265466655929225.jpg",
    PATH_NAME + "/train/data/044b17dca5212ca3.jpg"]


def main():
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(416, 416))

    # Visualize all preprocessing steps
    preprocessor.visualize_preprocessing_steps(image_path)

    # Visualize color channels
    preprocessor.visualize_color_channels(image_path)

    # Visualize aspect ratio handling
    preprocessor.visualize_aspect_ratio_handling(image_path)

    # Example batch visualization
    processed_batch, _ = preprocessor.process_batch(image_paths)
    preprocessor.visualize_batch_results(processed_batch)


def download_images():
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
    # Run this only once to download the images
    # download_images()
    main()
