from src.config import MODEL_PATH, BATCH_SIZE, LR, TRAIN_DIR, CLASSES
from src.train import train_model, plot_results, evaluate_model
from pathlib import Path
import tensorflow as tf
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Project root ko path mein add karo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def verify_system():
    """Training se pehle folders aur data check karta hai."""
    print("🔍 Running System Pre-Check...")
    if not TRAIN_DIR.exists():
        print(f"❌ Error: Train directory nahi mili: {TRAIN_DIR}")
        return False

    for cls in CLASSES:
        cls_path = TRAIN_DIR / cls
        if not cls_path.exists() or len(list(cls_path.glob('*'))) < 10:
            print(
                f"⚠️  Warning: Class '{cls}' mein images kam hain ya folder missing hai.")

    return True


def main():
    print("\n" + "="*55)
    print("🏛️  MONUMENT AI: MULTI-MODAL GPU SYSTEM (V5-REFINED) 🚀")
    print("="*55)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"✅ SUCCESS: NVIDIA GPU Detected ({len(gpus)} device(s))")
            print(f"ℹ️  Hardware  : Running on RTX 3050 with CUDA 11.2")
            print(f"ℹ️  Optimizer : Batch Size={BATCH_SIZE} | Initial LR={LR}")
        except RuntimeError as e:
            print(f"❌ GPU Config Error: {e}")
    else:
        print("⚠️  WARNING: GPU NOT detected. Running on CPU (Very Slow!).")

    # Pre-check logic
    if not verify_system():
        print("🛑 System check failed. Please fix data paths.")
        return

    try:

        history, val_gen = train_model()

        print("\n" + "-"*35)
        print("✅ Training complete! Plotting Metrics...")
        print("-"*35)

        if history:
            plot_results(history)

        if os.path.exists(MODEL_PATH):
            print(f"\n📦 Loading BEST saved model: {MODEL_PATH}")
            best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            best_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            evaluate_model(best_model, val_gen)
        else:
            print("❌ Error: Model file save nahi ho payi.")

        print("\n" + "="*55)
        print("🚀 System Execution Finished Successfully.")
        print(f"📂 Final Model: {MODEL_PATH}")
        print("="*55)

    except Exception as e:
        print(f"\n❌ Critical Execution Error: {e}")
        print("ℹ️  Tip: Batch size kam karke dekho agar OOM error aaye.")


if __name__ == "__main__":
    main()
