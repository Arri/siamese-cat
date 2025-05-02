import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
import tensorflow as tf
import logging
from sklearn.utils import shuffle


def resize_image(image, width=None, height=None):
    """Resize an image while keeping its aspect ratio."""
    (h, w) = image.shape[:2]

    # if width is None and height is None:
    #     return image  # No resizing needed
    sides = max(h, w)

    if width is not None:
        ratio = width / float(w)
        new_dim = (width, int(h * ratio))
    else:
        ratio = height / float(h)
        new_dim = (int(w * ratio), height)

    resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return resized


def resize_and_crop_square(img, sidelength):
    """
    Resizes an OpenCV image so that the smaller side becomes 'sidelength' while maintaining the aspect ratio,
    then crops the larger side to create a square image.

    :param img: OpenCV image (numpy array).
    :param sidelength: The desired side length of the square output image.
    :return: The cropped and resized OpenCV image.
    """
    height, width, _ = img.shape

    # Scale the image so that the smaller dimension becomes 'sidelength'
    if width < height:
        new_width = sidelength
        new_height = int((height / width) * sidelength)
    else:
        new_height = sidelength
        new_width = int((width / height) * sidelength)

    # Resize while keeping the aspect ratio
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Compute cropping coordinates to center-crop the larger dimension
    left = (new_width - sidelength) // 2
    top = (new_height - sidelength) // 2
    right = left + sidelength
    bottom = top + sidelength

    # Crop the image to the final square shape
    img_cropped = img_resized[top:bottom, left:right]

    return img_cropped



def compare_pr_curve_with_quantized_model(model_name, test_ds, output_dir):
    # Load full precision model
    embedding_model = tf.keras.models.load_model(
        f"embedding_{model_name}.keras", compile=False
    )

    # Load quantized model
    interpreter = tf.lite.Interpreter(model_path=f"embedding_{model_name}_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def run_quant_model(img_batch):
        results = []
        for img in img_batch:
            input_tensor = np.expand_dims(img.numpy(), axis=0)
            input_tensor = (input_tensor * 255).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            results.append(output[0])
        return np.array(results)

    anchor_embeddings_fp, positive_embeddings_fp, negative_embeddings_fp = [], [], []
    anchor_embeddings_q, positive_embeddings_q, negative_embeddings_q = [], [], []

    for (anchor, positive, negative), _ in test_ds:
        a_emb_fp = embedding_model.predict(anchor, verbose=0)
        p_emb_fp = embedding_model.predict(positive, verbose=0)
        n_emb_fp = embedding_model.predict(negative, verbose=0)

        a_emb_q = run_quant_model(anchor)
        p_emb_q = run_quant_model(positive)
        n_emb_q = run_quant_model(negative)

        anchor_embeddings_fp.append(a_emb_fp)
        positive_embeddings_fp.append(p_emb_fp)
        negative_embeddings_fp.append(n_emb_fp)

        anchor_embeddings_q.append(a_emb_q)
        positive_embeddings_q.append(p_emb_q)
        negative_embeddings_q.append(n_emb_q)

    def compute_scores(anchor, positive, negative):
        pos_dists = np.linalg.norm(anchor - positive, axis=1)
        neg_dists = np.linalg.norm(anchor - negative, axis=1)
        y_true = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
        scores = np.concatenate([-pos_dists, -neg_dists])
        return shuffle(scores, y_true, random_state=42)

    scores_fp, y_true_fp = compute_scores(
        np.vstack(anchor_embeddings_fp),
        np.vstack(positive_embeddings_fp),
        np.vstack(negative_embeddings_fp)
    )

    scores_q, y_true_q = compute_scores(
        np.vstack(anchor_embeddings_q),
        np.vstack(positive_embeddings_q),
        np.vstack(negative_embeddings_q)
    )

    prec_fp, rec_fp, _ = precision_recall_curve(y_true_fp, scores_fp)
    prec_q, rec_q, _ = precision_recall_curve(y_true_q, scores_q)

    plt.figure(figsize=(6, 5))
    plt.plot(rec_fp, prec_fp, label='Full Precision', marker='.')
    plt.plot(rec_q, prec_q, label='Quantized', marker='x')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'precision_recall_comparison.png'))
    plt.close()
