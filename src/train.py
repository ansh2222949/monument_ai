import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight

# Internal Project Imports
from .config import (
    TRAIN_DIR, TEST_DIR, BATCH_SIZE, EPOCHS,
    LR, MODEL_PATH, CLASSES
)
from .dataset import MonumentGenerator
from .model import build_monument_model

# =====================================================
# 🔍 EVALUATION FUNCTION
# =====================================================


def evaluate_model(model, val_gen):
    """Computes Accuracy, F1-score, and Detailed Report"""
    print("\n📊 Running Final Evaluation on Validation Set...")

    y_true, y_pred = [], []

    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(y_batch)
        y_pred.extend(preds)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("\n" + "⭐" * 35)
    print(f"✅ FINAL ACCURACY  : {acc:.4f} (90% Target Goal)")
    print(f"✅ F1-SCORE (MACRO) : {f1_macro:.4f}")
    print("⭐" * 35)

    print("\n📄 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

# =====================================================
# 🧠 TRAINING PIPELINE
# =====================================================


def train_model():
    print("📂 Preparing Data Generators...")
    train_gen = MonumentGenerator(
        TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = MonumentGenerator(
        TEST_DIR, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    labels = train_gen.labels
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(weights))

    print("🧠 Building Residual Multi-Modal Model from Scratch...")
    model = build_monument_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"🚀 Launching GPU Training for {EPOCHS} Epochs on GPU...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=callbacks,
        verbose=1
    )

    return history, val_gen

# =====================================================
# 📈 PLOTTING
# =====================================================


def plot_results(history):
    acc = history.history["accuracy"]
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history["loss"]
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy", color='blue', linewidth=2)
    plt.plot(val_acc, label="Validation Accuracy", color='orange', linewidth=2)
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% Target')
    plt.title("Monument AI: Accuracy Trend")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss", color='blue', linewidth=2)
    plt.plot(val_loss, label="Validation Loss", color='orange', linewidth=2)
    plt.title("Monument AI: Loss Trend")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
