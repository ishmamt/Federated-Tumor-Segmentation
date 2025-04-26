import matplotlib.pyplot as plt

def show_image_mask_tensor_pair(image,mask):
    # Remove channel dimension: (1, H, W) -> (H, W)
    image_np = image.squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy()

    # Plot side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Grayscale Image")
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_np, cmap='gray')  # Or use cmap='Reds' for binary masks
    plt.axis('off')

    plt.tight_layout()
    plt.show()