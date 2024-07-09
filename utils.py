import numpy as np
import matplotlib.pyplot as plt


def show_image(img, msk, labels=['mask'], semantic=False, threshold=False):
    fig, axs = plt.subplots(1, 1 + len(labels), figsize=(8, 3))
    img = np.squeeze(img)

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("image")

    for i in range(len(labels)):
        if semantic and threshold:
            # Create a binary mask based on the channel with the highest value
            binary_mask = np.argmax(msk, axis=0) == i
            axs[i + 1].imshow(binary_mask, vmin=0, vmax=1, cmap='gray')
        else:
            axs[i + 1].imshow(msk[i], vmin=0, vmax=1, cmap='gray')
        axs[i + 1].set_title(f"{labels[i]}")

    plt.show()