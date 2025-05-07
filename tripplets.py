import random
import logging 
import tensorflow as tf
from prep_data import group_by_label


def decode_image(path, imgsize):
    ''' Read the image file, decode jpeg and resize the image to the size 
    required by the model, normalize the image to [0, 1] '''
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [imgsize, imgsize])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_image(img):
    ''' Only during training (augment=True) apply random augmentations
    (flip left/right, change brightness and contrast)'''
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img


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


def process_triplet(anchor_path, pos_path, neg_path, imgsize, augment=False):
    ''' Read, prepare and augment images, ready for the training.
    Load and preprocess the three images in a triplet. '''
    a = decode_image(anchor_path, imgsize)
    p = decode_image(pos_path, imgsize)
    n = decode_image(neg_path, imgsize)
    if augment:
        a = augment_image(a)
        p = augment_image(p)
        n = augment_image(n)
    return (a, p, n), tf.constant(1.0)


def create_triplet_dataset(triplets, batch_size, imgsize,
                           augment=False, shuffle=True):
    ''' Construct a TensorFlow dataset suitable for training a Siamese or
    Triplet Network.
    This returns a batched and preprocessed `tf.data.Dataset` of triplets
    ready for training a model with triplet loss.'''
    anchor_paths, pos_paths, neg_paths = zip(*triplets)
    ds = tf.data.Dataset.from_tensor_slices((list(anchor_paths),
                                             list(pos_paths),
                                             list(neg_paths)))
    ds = ds.map(lambda a, p, n: process_triplet(a, p, n, imgsize, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
