import argparse
from datetime import datetime
import os

import numpy as np
import tensorflow as tf
import logging

from display import plot_split_images
from utils import compare_pr_curve_with_quantized_model
from model import embedding_model_tl, complete_model_tl
from evaluate import evaluate_model
from quantize import quantize_model
from prep_data import filter_rare_classes, split_train_val_test
from tripplets import generate_triplets, create_triplet_dataset

# from collections import defaultdict, Counter
# from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model


np.random.seed(2024)
IMG_SIZE = 224   # 240
batch_size = 16
num_epochs = 50
learning_rate = 0.0001


def collect_image_paths(base_dir='hsc_dataset', photo_type='front'):
    ''' Scans a dataset directory and returns a list of image file
    paths along with their corresponding labels. '''
    assert photo_type in ["front", "top", "all"]
    image_paths, labels = [], []
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        label = entry.split("-")[-1] if "-" in entry else None
        if label is None:
            continue
        subfolders = ["front", "top"] if photo_type == "all" else [photo_type]
        for subfolder in subfolders:
            folder_path = os.path.join(entry_path, subfolder)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        image_paths.append(os.path.join(folder_path, fname))
                        labels.append(label)
    print(f"[INFO] Amount of imagees before filtering = {len(image_paths)}")
    logging.info(f"Amount of imagees before filtering = {len(image_paths)}")
    return image_paths, labels


def run_single_inference(reference_img_path, test_img_path,
                         model_name, threshold=0.5):
    embedding_model = tf.keras.models.load_model(
        f"embedding_{model_name}.keras", compile=False
    )

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return tf.expand_dims(img, axis=0)

    ref_img = load_and_preprocess(reference_img_path)
    test_img = load_and_preprocess(test_img_path)

    ref_emb = embedding_model.predict(ref_img, verbose=0)
    test_emb = embedding_model.predict(test_img, verbose=0)

    distance = np.linalg.norm(ref_emb - test_emb)
    is_same = distance < threshold

    print(f"[INFO] Distance: {distance:.4f} | Same animal: {is_same}")
    return is_same


def main(model_name, evaluate_only=False, num_samples=100,
         fine_tune_layers=20, fine_tune_epochs=5, margin=None):
    paths, labels = collect_image_paths(photo_type="front")
    paths, labels = filter_rare_classes(paths, labels, min_count=2)
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)

    (train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
     ) = split_train_val_test(paths, numeric_labels)

    plot_split_images(train_paths, train_labels, IMG_SIZE, label_encoder=le,
                      n_rows=4, n_cols=6, title="Training Set Examples")
    plot_split_images(test_paths, test_labels, IMG_SIZE, label_encoder=le,
                      n_rows=4, n_cols=6, title="Testing Set Examples")

    output_dir = (f"results_{model_name}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_dir, 'evaluation.log'),
                        level=logging.INFO)

    if not evaluate_only:
        train_triplets = generate_triplets(train_paths, train_labels,
                                           "Training",
                                           num_triplets_per_class=100,
                                           )
        val_triplets = generate_triplets(val_paths, val_labels,
                                         "Validation",
                                         num_triplets_per_class=40,
                                         )

        train_ds = create_triplet_dataset(train_triplets, batch_size,
                                          IMG_SIZE, augment=True)
        val_ds = create_triplet_dataset(val_triplets, batch_size,
                                        IMG_SIZE, augment=False)

        emb_model = embedding_model_tl((IMG_SIZE, IMG_SIZE, 3),
                                       b_model=model_name,
                                       embeddingDim=128,
                                       unfreeze_layers=fine_tune_layers)
        model = complete_model_tl(emb_model, IMG_SIZE, learning_rate)

        siamese_model_name = f"siamese_{model_name}_best.keras"
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            siamese_model_name,
            save_best_only=True, monitor='val_loss'
        )
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max')

        logging.info("========================= TRAINING PHASE "
                     "=========================")
        logging.info("Starting training...")
        print("[INFO] Starting training...")
        # Log model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        for line in summary_lines:
            logging.info(line)
            print(line)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            callbacks=[checkpoint_cb, early_stop_cb]
        )
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                         f"loss: {history.history['loss'][epoch]:.4f},"
                         " val_loss: "
                         f"{history.history['val_loss'][epoch]:.4f}")

        # Phase 2: Fine-tune by unfreezing more base model layers
        print("Fine-tuning the base model...")
        for layer in emb_model.layers:
            layer.trainable = True

        model = complete_model_tl(emb_model, IMG_SIZE, LR=learning_rate * 0.1)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=[checkpoint_cb, early_stop_cb]
        )
        for epoch in range(fine_tune_epochs):
            logging.info(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs} "
                         f"- loss: {history_ft.history['loss'][epoch]:.4f}, "
                         "val_loss: "
                         f"{history_ft.history['val_loss'][epoch]:.4f}")

        # Call function to quantize the model:
        quantize_model(model_name, siamese_model_name, train_ds, emb_model)

    logging.info("========================= EVALUATION PHASE "
                 "=========================")
    print("============== Evaluation Phase ================")
    print("[INFO] Evaluating on test set...")
    test_triplets = generate_triplets(test_paths, test_labels,
                                      "Testing",
                                      num_triplets_per_class=100)
    test_ds = create_triplet_dataset(test_triplets, batch_size, IMG_SIZE,
                                     augment=False, shuffle=False)
    evaluate_model(model_name=model_name, test_ds=test_ds,
                   output_dir=output_dir, margin=margin)
    compare_pr_curve_with_quantized_model(model_name, test_ds, output_dir)


if __name__ == "__main__":
    # CLI parsing and execution
    parser = argparse.ArgumentParser(
        description="Train Siamese network with triplet loss")
    parser.add_argument(
        "--model",
        type=str,
        choices=['vgg', 'efficient', 'mobile', 'mobileV2', 'MobileNetV3Small',
                 'MobileNetV3Large'],
        default="mobileV2",
        help="Specify the model to use"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Only evaluate the model without training"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of positive pairs to generate per class for evaluation"
    )
    parser.add_argument(
        "--fine_tune_layers",
        type=int,
        default=20,
        help="Number of base model layers to unfreeze during fine-tuning"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="""Override the learned threshold with a
        custom margin for evaluation (in distance units)"""
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=5,
        help="Number of epochs for the fine-tuning phase"
    )

    args = parser.parse_args()
    main(model_name=args.model, evaluate_only=args.evaluate,
         num_samples=args.num_samples,
         fine_tune_layers=args.fine_tune_layers,
         fine_tune_epochs=args.fine_tune_epochs,
         margin=args.margin)
