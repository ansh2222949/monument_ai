import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
from .config import IMG_SIZE, BATCH_SIZE, CLASSES


def apply_balanced_augmentation(img):
    """
    Refined Augmentation: Architecture-focused transforms without over-distorting.
    """
    h, w = img.shape[:2]

    # 1. Horizontal Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Subtle Perspective Transform
    if random.random() > 0.4:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = random.uniform(0, 0.10) * w
        pts2 = np.float32([[offset, 0], [w-offset, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(
            img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Rotation & Shear (Slight variations)
    angle = random.uniform(-10, 10)
    shear = random.uniform(-0.05, 0.05)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    M[:, 0] += shear
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 4. Zoom Jitter (0.95x to 1.05x)
    zoom = random.uniform(0.95, 1.05)
    img = cv2.resize(img, None, fx=zoom, fy=zoom)
    new_h, new_w = img.shape[:2]

    if zoom > 1.0:
        img = img[(new_h-h)//2:(new_h-h)//2+h, (new_w-w)//2:(new_w-w)//2+w]
    else:
        pad_h, pad_w = (h-new_h)//2, (w-new_w)//2
        img = cv2.copyMakeBorder(
            img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_REPLICATE)

    # 5. Brightness & Blur
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() > 0.8:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def get_views(img_path, augment=False):
    """
    1 base photo se 4 synchronized representations nikalna.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Synchronized Augmentation
        if augment:
            img = apply_balanced_augmentation(img)

        # ---------- GRAYSCALE ----------
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])

        # ---------- EDGE ----------
        edges = cv2.Canny(gray, 100, 200)
        edges_3ch = cv2.merge([edges, edges, edges])

        # ---------- PSEUDO DEPTH ----------
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        depth = np.uint8(np.absolute(lap))
        depth_3ch = cv2.merge([depth, depth, depth])

        return (
            img.astype(np.float32) / 255.0,
            depth_3ch.astype(np.float32) / 255.0,
            gray_3ch.astype(np.float32) / 255.0,
            edges_3ch.astype(np.float32) / 255.0
        )
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return None


class MonumentGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.image_paths = []
        self.labels = []

        for idx, label in enumerate(CLASSES):
            class_dir = os.path.join(directory, label)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index *
                                     self.batch_size: (index + 1) * self.batch_size]
        batch_rgb, batch_depth, batch_gray, batch_edge, batch_y = [], [], [], [], []

        for i in batch_indexes:
            views = get_views(self.image_paths[i], augment=self.augment)
            if views is None:
                continue

            rgb, depth, gray, edge = views
            batch_rgb.append(rgb)
            batch_depth.append(depth)
            batch_gray.append(gray)
            batch_edge.append(edge)
            batch_y.append(self.labels[i])

        return (
            [
                np.array(batch_rgb, dtype=np.float32),
                np.array(batch_depth, dtype=np.float32),
                np.array(batch_gray, dtype=np.float32),
                np.array(batch_edge, dtype=np.float32),
            ],
            np.array(batch_y, dtype=np.int32),
        )
