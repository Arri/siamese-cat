import argparse
from datetime import datetime
import os
import random
import numpy as np
import tensorflow as tf
import logging

from display import plot_split_images
from utils import compare_pr_curve_with_quantized_model
from model import embedding_model_tl, complete_model_tl
from evaluate import evaluate_model

from collections import defaultdict, Counter
# from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
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


def filter_rare_classes(paths, labels, min_count=2):
    ''' Collect only those samples that have at least 
    min_count (=2) images from the same cat/dog etc.'''
    label_counts = Counter(labels)
    filtered_paths, filtered_labels = [], []
    for path, label in zip(paths, labels):
        if label_counts[label] >= min_count:
            filtered_paths.append(path)
            filtered_labels.append(label)
    print(f"[INFO] Amount of images after filtering rare cases = {len(filtered_paths)}")
    logging.info(f"Amount of images after filtering rare cases = {len(filtered_paths)}")
    return filtered_paths, filtered_labels


def split_train_val_test(paths, labels, val_size=0.1, test_size=0.1, seed=42):
    ''' Using Scikit-Learn split the paths of the original dataset
    into a training, a validation and a testing set. '''
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, random_state=seed
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_size / (1.0 - test_size),
        random_state=seed
    )
    print(f"[INFO] Training samples = {len(train_paths)}")
    logging.info(f"Training samples = {len(train_paths)}")
    print(f"[INFO] Validation samples = {len(val_paths)}")
    logging.info(f"Validation samples = {len(val_paths)}")
    print(f"[INFO] Testing samples = {len(test_paths)}")
    logging.info(f"Testing samples = {len(test_paths)}")
    return (train_paths, val_paths, test_paths,
            train_labels, val_labels, test_labels)


def group_by_label(paths, labels):
    ''' Orgnize a list of file paths by their corresponding labels.
    Returns a dictionary-like object mapping each label to a list of all file
    paths with that label '''
    label_to_paths = defaultdict(list)
    for path, label in zip(paths, labels):
        label_to_paths[label].append(path)
    return label_to_paths


def generate_triplets(paths, labels, name, num_triplets_per_class=50):
    ''' The returned triplets in a list containing N tuples with
    the paths to three samples: One for the Anchor, one for the
    Positive sample (e.g. same cat as the anchor) and one for the
    Negative sample. '''
    label_to_paths = group_by_label(paths, labels)
    all_labels = list(label_to_paths.keys())
    triplets = []
    for label in all_labels:
        pos_images = label_to_paths[label]
        if len(pos_images) < 2:
            continue
        for _ in range(num_triplets_per_class):
            anchor, positive = random.sample(pos_images, 2)

            neg_label = random.choice(
                [lab for lab in all_labels if lab != label])
            negative = random.choice(label_to_paths[neg_label])
            triplets.append((anchor, positive, negative))
    print(f"[INFO] Amount of tripplets for {name} = {len(triplets)}.")
    logging.info(f"Amount of tripplets for {name} = {len(triplets)}.")
    return triplets


def decode_image(path):
    ''' Read the image file, decode jpeg and resize the image to the size 
    required by the model, normalize the image to [0, 1] '''
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_image(img):
    ''' Only during training (augment=True) apply random augmentations
    (flip left/right, change brightness and contrast)'''
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img


def process_triplet(anchor_path, pos_path, neg_path, augment=False):
    ''' Read, prepare and augment images, ready for the training.
    Load and preprocess the three images in a triplet. '''
    a = decode_image(anchor_path)
    p = decode_image(pos_path)
    n = decode_image(neg_path)
    if augment:
        a = augment_image(a)
        p = augment_image(p)
        n = augment_image(n)
    return (a, p, n), tf.constant(1.0)


def create_triplet_dataset(triplets, augment=False, shuffle=True):
    ''' Construct a TensorFlow dataset suitable for training a Siamese or
    Triplet Network.
    This returns a batched and preprocessed `tf.data.Dataset` of triplets
    ready for training a model with triplet loss.'''
    anchor_paths, pos_paths, neg_paths = zip(*triplets)
    ds = tf.data.Dataset.from_tensor_slices((list(anchor_paths),
                                             list(pos_paths),
                                             list(neg_paths)))
    ds = ds.map(lambda a, p, n: process_triplet(a, p, n, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


def main(model_name, evaluate_only=False, num_samples=100, fine_tune_layers=20, fine_tune_epochs=5, margin=None):
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

    output_dir = f"results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_dir, 'evaluation.log'), level=logging.INFO)

    if not evaluate_only:
        train_triplets = generate_triplets(train_paths, train_labels,
                                           "Training",
                                           num_triplets_per_class=100,
                                           )
        val_triplets = generate_triplets(val_paths, val_labels,
                                         "Validation",
                                         num_triplets_per_class=40,
                                         )

        train_ds = create_triplet_dataset(train_triplets, augment=True)
        val_ds = create_triplet_dataset(val_triplets, augment=False)

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

        logging.info("========================= TRAINING PHASE =========================")
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
            logging.info(f"Epoch {epoch+1}/{num_epochs} - loss: {history.history['loss'][epoch]:.4f}, val_loss: {history.history['val_loss'][epoch]:.4f}")

        # Phase 2: Fine-tune by unfreezing more base model layers
        print("Fine-tuning base model...")
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
            logging.info(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs} - loss: {history_ft.history['loss'][epoch]:.4f}, val_loss: {history_ft.history['val_loss'][epoch]:.4f}")

        # Save the trained embedding model AFTER training
        embedding_model_name = f"embedding_{model_name}.keras"
        emb_model.save(embedding_model_name)
        print("[INFO] Training complete.")
        print(f"\tBest model saved as {siamese_model_name}.")
        logging.info(f"Best model saved as {siamese_model_name}.")
        print(f"\tThe embedding model was saved as {embedding_model_name}.")
        logging.info(f"The embedding model was saved as {embedding_model_name}.")

        # Save quantized model
        logging.info("========================= Quantization PHASE =========================")
        print("[INFO] Evaluating on test set...")

        def representative_data_gen():
            for anchor, _, _ in train_ds.take(100):
                yield [anchor]

        converter = tf.lite.TFLiteConverter.from_keras_model(emb_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()
        quantized_model_file = f"embedding_{model_name}_int8.tflite"
        with open(quantized_model_file, "wb") as f:
            f.write(tflite_quant_model)
        logging.info(f"Converted the model to 8-bit. Quantized model saved as {quantized_model_file}.")
        print(f"[INFO] 8-bit quantized model saved as {quantized_model_file}.")

    logging.info("========================= EVALUATION PHASE =========================")
    print("[INFO] Evaluating on test set...")
    test_triplets = generate_triplets(test_paths, test_labels,
                                      "Testing",
                                      num_triplets_per_class=100)
    test_ds = create_triplet_dataset(test_triplets,
                                     augment=False, shuffle=False)
    evaluate_model(model_name=model_name, test_ds=test_ds, output_dir=output_dir, margin=margin)
    compare_pr_curve_with_quantized_model


if __name__ == "__main__":
    # CLI parsing and execution
    parser = argparse.ArgumentParser(
        description="Train Siamese network with triplet loss")
    parser.add_argument(
        "--model",
        type=str,
        choices=['vgg', 'efficient', 'mobile', 'mobileV2', 'MobileNetV3Small', 'MobileNetV3Large'],
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
        help="Override the learned threshold with a custom margin for evaluation (in distance units)"
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
