# 🏛️ Monument AI — Multi-Modal Monument Recognition (From Scratch)

Monument AI is a **research-oriented deep learning project** that performs **monument recognition from images** using a **custom multi-modal CNN built completely from scratch**.

Instead of relying on a single RGB image, the model learns **complementary structural and visual representations** (RGB, grayscale, depth, and edge views) through **parallel residual CNN branches**, making it more robust and interpretable on limited data.

> ⚠️ This project intentionally avoids pretrained models to focus on **architecture design, learning stability, and reasoning**, not leaderboard chasing.

---

## ✨ Key Features

- 🧠 **Multi-Modal Learning**
  - RGB (appearance)
  - Grayscale (texture & lighting invariance)
  - Depth (structural geometry)
  - Edge maps (shape & contours)

- 🏗️ **Custom Residual CNN (From Scratch)**
  - Skip connections for stable gradient flow
  - No pretrained backbones

- ⚖️ **Class Imbalance Handling**
  - Explicit class weighting
  - Macro F1-score evaluation

- 📊 **Robust Evaluation**
  - Accuracy + Macro F1
  - Detailed per-class classification report

- 🖥️ **Desktop GUI Demo**
  - Drag & drop image inference
  - Visual display of all 4 input modalities
  - Confidence bar & predictions

---

## 🧠 Why Multi-Modal CNN?

Monuments often share similar visual patterns (arches, walls, symmetry), making single-representation models brittle.

This project injects **inductive bias** by separating learning into specialized branches:

- **RGB** → overall appearance
- **Grayscale** → texture robustness
- **Depth** → structural layout
- **Edges** → geometric shape

Each branch learns independently, and their features are **fused** for final classification.

This design improves:
- Learning stability on small datasets
- Structural understanding
- Explainability of predictions

---

## 🏗️ Model Architecture (High-Level)

```text
RGB Image ──┐
Depth Map ─┼──▶ Residual CNN Branches ─▶ Feature Fusion ─▶ Classifier ─▶ Monument
Gray Image ─┤
Edge Map ──┘
```
Residual blocks mitigate vanishing gradients

Global Average Pooling reduces overfitting

Dense head balances capacity and regularization



```text
MONUMENT_AI/
├── data/
│   ├── train/                # Training images (class-wise folders)
│   └── test/                 # Validation / unseen images
│
├── outputs/
│   └── best_monument_model.h5
│
├── src/
│   ├── config.py             # Paths, hyperparameters, class names
│   ├── dataset.py            # Data loader + multi-view generation
│   ├── model.py              # Multi-modal residual CNN architecture
│   └── train.py              # Training & evaluation pipeline
│
├── gui.py                    # Desktop GUI for inference & visualization
├── predict.py                # CLI / single-image inference
├── run.py                    # System & GPU sanity checks
├── requirements.txt
└── README.md
```
📊 Training Strategy

Optimizer: Adam

Loss: Sparse Categorical Cross-Entropy

Regularization:

Dropout

Early stopping

Learning rate reduction on plateau

Best model selection based on validation loss

Class imbalance is explicitly handled using balanced class weights.

📈 Evaluation Metrics

Accuracy

Macro F1-Score (preferred due to class imbalance)

Full classification report (precision / recall / F1 per class)

Macro F1 is emphasized to ensure fair performance across all monuments.

🖥️ GUI Demo

The project includes a desktop GUI that:

Accepts drag & drop images

Displays all 4 modality views (RGB, depth, gray, edges)

Shows predicted monument & confidence

This tool is intended for qualitative analysis and explainability, not production deployment.

⚠️ Limitations

Dataset size is limited

Depth maps are approximated (not sensor-grade)

Not optimized for real-time deployment

Windows-focused tooling

These are intentional trade-offs to prioritize learning and experimentation.

🚀 Future Work

Attention-based feature fusion

Better depth estimation models

Ablation study (single vs multi-modal)

Lightweight web inference interface (separate project)

🧠 Key Takeaway

Monument AI is not about achieving the highest accuracy —
it is about understanding how different visual representations contribute to recognition and building a clean, explainable deep learning system from scratch.

📜 License

This project is intended for educational and research purposes.