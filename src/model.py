import tensorflow as tf
from tensorflow.keras import layers, models
from .config import IMG_SIZE, NUM_CLASSES


def residual_block(x, filters):
    """
    Refined Residual Block: Information flow ko smooth rakhta hai.
    Vanishing gradient rokne ke liye skip connections best hain.
    """
    shortcut = x

    # Path 1: Convolutional layers
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), padding="same", kernel_initializer="he_normal")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def create_pro_scratch_branch(input_shape, name_prefix):
    """
    Optimized branch for multimodal views (RGB, Depth, Gray, Edge).
    """
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")

    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_initializer="he_normal", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 64)

    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)

    return inputs, x


def build_monument_model():
    """
    Monument AI: Multi-modal Residual Architecture tuned .
    """

    # 1️⃣ Parallel Refined Branches
    rgb_input, rgb_feat = create_pro_scratch_branch(
        (IMG_SIZE, IMG_SIZE, 3), "rgb")
    depth_input, depth_feat = create_pro_scratch_branch(
        (IMG_SIZE, IMG_SIZE, 3), "depth")
    gray_input, gray_feat = create_pro_scratch_branch(
        (IMG_SIZE, IMG_SIZE, 3), "gray")
    edge_input, edge_feat = create_pro_scratch_branch(
        (IMG_SIZE, IMG_SIZE, 3), "edge")

    # 2️⃣ Feature Fusion
    fused = layers.Concatenate(name="feature_fusion")(
        [rgb_feat, depth_feat, gray_feat, edge_feat]
    )

    # 3️⃣ Balanced Classification Head

    x = layers.Dense(512, kernel_initializer="he_normal")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(
        inputs=[rgb_input, depth_input, gray_input, edge_input],
        outputs=outputs,
        name="MonumentAI_Residual_v5_Refined"
    )

    return model
