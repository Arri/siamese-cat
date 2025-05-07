import logging
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split


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