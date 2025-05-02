import os
import numpy as np
import logging
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve

from display import plot_image

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_name, test_ds, output_dir=None, margin=None):
    print(f"[INFO] Loading trained embedding model from embedding_{model_name}.keras")
    logging.info(f"Loading trained embedding model from embedding_{model_name}.keras")
    embedding_model = tf.keras.models.load_model(
        f"embedding_{model_name}.keras", compile=False
    )

    all_anchor_embeddings = []
    all_positive_embeddings = []
    all_negative_embeddings = []

    # Iterate through the test dataset
    for (anchor, positive, negative), _ in test_ds:
        # Get the embeddings for each image type separately
        a_emb = embedding_model.predict(anchor, verbose=0)
        p_emb = embedding_model.predict(positive, verbose=0)
        n_emb = embedding_model.predict(negative, verbose=0)

        # Append the embeddings to the corresponding lists
        all_anchor_embeddings.append(a_emb)
        all_positive_embeddings.append(p_emb)
        all_negative_embeddings.append(n_emb)

    # Stack the embeddings for further processing
    anchor_embeddings = np.vstack(all_anchor_embeddings)
    positive_embeddings = np.vstack(all_positive_embeddings)
    negative_embeddings = np.vstack(all_negative_embeddings)

    # Compute distances
    pos_dists = np.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
    neg_dists = np.linalg.norm(anchor_embeddings - negative_embeddings, axis=1)
    print(f"[INFO] positive distance = {pos_dists} --- length = {len(pos_dists)}")
    logging.info(f"positive distance = {pos_dists} --- length = {len(pos_dists)}")
    print(f"[INFO] negative distance = {neg_dists} --- length = {len(neg_dists)}")
    logging.info(f"negative distance = {neg_dists} --- length = {len(neg_dists)}")

    # Predict: 1 if pos_dist < neg_dist (i.e., positive closer)
    # y_true = np.ones_like(pos_dists)
    y_true = np.concatenate([
        np.ones_like(pos_dists),     # label 1 for positives
        np.zeros_like(neg_dists)     # label 0 for negatives
    ])

    # Concatenate distances as similarity scores (invert since smaller = more similar)
    scores = np.concatenate([
        -pos_dists,  # more similar → higher score
        -neg_dists   # less similar → lower score
    ])

    # Optional: shuffle to remove ordering bias
    scores, y_true = shuffle(scores, y_true, random_state=42)

    # Apply a default threshold for binary prediction
    # Auto-tune threshold using F1 score
    thresholds = np.linspace(min(scores), max(scores), 200)
    best_f1 = 0.0
    best_threshold = 0.0
    for t in thresholds:
        preds = (scores > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    y_pred = (scores > best_threshold).astype(int)
    print(f"[INFO] Auto-tuned threshold: {best_threshold:.4f} (Best F1: {best_f1:.4f})")
    logging.info(f"Auto-tuned threshold: {best_threshold:.4f} (Best F1: {best_f1:.4f})")

    # Plot F1 vs threshold
    f1_scores = [f1_score(y_true, (scores > t).astype(int)) for t in thresholds]
    filename = os.path.join(output_dir, 'f1_vs_threshold.png')
    plot_image(thresholds, f1_scores, 'Threshold',  'F1 Score', 'F1 Score vs. Threshold', filename)

    # Plot histogram of distances
    plt.figure(figsize=(6, 5))
    plt.hist(pos_dists, bins=50, alpha=0.5, label='Positive distances')
    plt.hist(neg_dists, bins=50, alpha=0.5, label='Negative distances')
    plt.axvline(-best_threshold, color='red', linestyle='--', label=f'Threshold = {-best_threshold:.2f}')
    plt.legend()
    plt.title('Distance Distributions')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'distance_distributions.png'))
    plt.show()

    # Calculate performance metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Combined absolute and percentual confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Print results
    print(f"[INFO] Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logging.info(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title('Confusion Matrix (Absolute)')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')

    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', ax=axs[1])
    axs[1].set_title('Confusion Matrix (%)')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.show()

    # Reuse shuffled scores and y_true for PR curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, scores)
    filename = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_image(rec_vals, prec_vals, 'Recall',  'Precision', 'Precision-Recall Curve', filename)
