import os
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf


def show_images(images, labels, n):
    """ Display the first n images in a list """
    cols = 4   # number of images per column
    rows = math.ceil(n / cols)  # Calculate the required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))    # Figure size
    axes = axes.flatten()       # Flatten axes for easy iteration

    for i, image in enumerate(images[:n]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"Image {labels[i]}")

    # Hide any unused subplot axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    return


def plot_image(x_vals, y_vals, xname, yname, titname, filename):
    plt.figure(figsize=(6, 5))
    plt.plot(x_vals, y_vals, marker='.')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(titname)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    return


def plot_split_images(paths, labels, imgsize, label_encoder=None,
                      n_rows=3, n_cols=5, title='Images'):
    assert len(paths) == len(labels)

    total = min(len(paths), n_rows * n_cols)
    selected_indices = np.random.choice(len(paths), total, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for idx, ax in zip(selected_indices, axes):
        img = tf.io.read_file(paths[idx])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [imgsize, imgsize]) / 255.0
        ax.imshow(img.numpy())
        label_text = label_encoder.inverse_transform([labels[idx]])[0] if label_encoder else labels[idx]
        ax.set_title(f"Label: {label_text}", fontsize=8)
        ax.axis('off')

    for ax in axes[total:]:
        ax.axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()