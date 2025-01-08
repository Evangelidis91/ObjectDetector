from OpenImagesDatasetPreparation import OpenImagesDatasetPreparation

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
