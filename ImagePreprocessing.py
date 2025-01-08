import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

"""
visualize_preprocessing_steps(), visualize_batch_results(), visualize_color_channels() and  visualize_aspect_ratio_handling
are just to visualize the result
"""


class ImagePreprocessor:
    def __init__(self, target_size=(416, 416)):
        """
                Initialize the image preprocessor
                Args:
                    target_size: Tuple of (height, width) for output images
                """
        self.target_size = target_size

    def resize_with_aspect_ratio(self, image):
        """
                Resize image while maintaining aspect ratio
                Args:
                    image: Input image array
                Returns:
                    Resized image with padding if necessary
                """
        ih, iw = image.shape[:2]
        h, w = self.target_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(h / ih, w / iw)

        # Calculate new dimension
        new_h = int(ih * scale)
        new_w = int(iw * scale)

        # Resize Image
        resized = cv2.resize(image, (new_w, new_h))

        # Create blank canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Calculate padding
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        # Place resized image on canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas, (x_offset, y_offset, scale)

    def normalize_image(self, image):
        """
                Normalize pixel values to range [0, 1]
                Args:
                    image: Input image array
                Returns:
                    Normalized image array
                """
        return image.astype(np.float32) / 255.0

    def handle_color_channels(self, image):
        """
        Ensure image has correct number of color channels
        Args:
            image: Input image array
        Returns:
            Image array with 3 color channels
        """
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def preprocess_image(self, image_path):
        """
        Complete preprocessing pipeline for a single image
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image and transformation info
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Handle color channels
            image = self.handle_color_channels(image)

            # Resize maintaining aspect ratio
            resized_image, transform_info = self.resize_with_aspect_ratio(image)

            # Normalize pixel values
            normalized_image = self.normalize_image(resized_image)

            return normalized_image, transform_info

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None, None

    def process_batch(self, image_paths, batch_size=32):
        """
        Process a batch of images
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
        Returns:
            Batch of preprocessed images and transformation info
        """
        processed_images = []
        transform_infos = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_transforms = []

            for path in batch_paths:
                image, transform = self.preprocess_image(path)
                if image is not None:
                    batch_images.append(image)
                    batch_transforms.append(transform)

            processed_images.extend(batch_images)
            transform_infos.extend(batch_transforms)

        return np.array(processed_images), transform_infos

    def visualize_preprocessing_steps(self, image_path):
        """
        Visualize each step of the preprocessing pipeline
        Args:
            image_path: Path to the image file
        """
        # Read original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Get preprocessed image
        processed, transform_info = self.preprocess_image(image_path)

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))

        # Create grid of images
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.3)

        # Original Image
        grid[0].imshow(original)
        grid[0].set_title('Original Image\n'
                          f'Shape: {original.shape}\n'
                          f'Value Range: [{original.min()}, {original.max()}]')

        # Resized Image (before normalization)
        resized, _ = self.resize_with_aspect_ratio(original)
        grid[1].imshow(resized)
        grid[1].set_title('Resized Image\n'
                          f'Shape: {resized.shape}\n'
                          f'Value Range: [{resized.min()}, {resized.max()}]')

        # Final Processed Image
        grid[2].imshow(processed)
        grid[2].set_title('Normalized Image\n'
                          f'Shape: {processed.shape}\n'
                          f'Value Range: [{processed.min():.2f}, {processed.max():.2f}]')

        # Remove axes
        for ax in grid:
            ax.axis('off')

        plt.suptitle(f'Preprocessing Steps for {image_path.split("/")[-1]}')
        plt.show()

    def visualize_batch_results(self, processed_batch, num_samples=5):
        """
        Visualize a sample of preprocessed images from a batch
        Args:
            processed_batch: Batch of preprocessed images
            num_samples: Number of samples to visualize
        """
        num_samples = min(num_samples, len(processed_batch))

        # Create figure
        fig = plt.figure(figsize=(15, 3 * num_samples))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(num_samples, 1),
                         axes_pad=0.3)

        for idx, ax in enumerate(grid):
            if idx < num_samples:
                ax.imshow(processed_batch[idx])
                ax.set_title(f'Sample {idx + 1}\n'
                             f'Shape: {processed_batch[idx].shape}\n'
                             f'Value Range: [{processed_batch[idx].min():.2f}, '
                             f'{processed_batch[idx].max():.2f}]')
                ax.axis('off')

        plt.suptitle('Batch Processing Results')
        plt.show()

    def visualize_color_channels(self, image_path):
        """
        Visualize individual color channels of the processed image
        Args:
            image_path: Path to the image file
        """
        processed, _ = self.preprocess_image(image_path)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original RGB
        axes[0].imshow(processed)
        axes[0].set_title('RGB Combined')

        # Individual channels
        channel_names = ['Red', 'Green', 'Blue']
        for i in range(3):
            channel = np.zeros_like(processed)
            channel[:, :, i] = processed[:, :, i]
            axes[i + 1].imshow(channel)
            axes[i + 1].set_title(f'{channel_names[i]} Channel\n'
                                  f'Mean: {processed[:, :, i].mean():.3f}\n'
                                  f'Std: {processed[:, :, i].std():.3f}')

        for ax in axes:
            ax.axis('off')

        plt.suptitle(f'Color Channel Analysis for {image_path.split("/")[-1]}')
        plt.show()

    def visualize_aspect_ratio_handling(self, image_path):
        """
        Visualize how aspect ratio is maintained with padding
        Args:
            image_path: Path to the image file
        """
        # Read original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Get preprocessed image with padding info
        processed, (x_offset, y_offset, scale) = self.resize_with_aspect_ratio(original)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Original image with aspect ratio lines
        ax1.imshow(original)
        ax1.set_title(f'Original Image\nAspect Ratio: {original.shape[1] / original.shape[0]:.2f}')

        # Processed image with padding visualization
        ax2.imshow(processed)
        ax2.set_title(f'Processed Image\nAspect Ratio: {processed.shape[1] / processed.shape[0]:.2f}\n'
                      f'Scale: {scale:.2f}')

        # Show padding areas
        if y_offset > 0:
            ax2.axhline(y=y_offset, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=processed.shape[0] - y_offset, color='r', linestyle='--', alpha=0.5)
        if x_offset > 0:
            ax2.axvline(x=x_offset, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=processed.shape[1] - x_offset, color='r', linestyle='--', alpha=0.5)

        for ax in (ax1, ax2):
            ax.axis('off')

        plt.suptitle('Aspect Ratio Handling')
        plt.show()
