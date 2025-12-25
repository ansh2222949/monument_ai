import os
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf

# Project Imports
from src.config import MODEL_PATH, CLASSES, IMG_SIZE
from src.dataset import get_views

# ================= THEME PALETTES =================
THEMES = {
    "Ocean Blue": {"accent": "#3B8ED0", "hover": "#1F538D"},
    "Forest Green": {"accent": "#2ECC71", "hover": "#27AE60"},
    "Crimson Red": {"accent": "#E74C3C", "hover": "#C0392B"},
    "Cyber Purple": {"accent": "#9B59B6", "hover": "#8E44AD"},
    "Slate Mono": {"accent": "#7F8C8D", "hover": "#34495E"}
}

# ================= HYBRID WRAPPER =================


class EliteWindow(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class MonumentAI_Elite_GUI(EliteWindow):
    def __init__(self):
        super().__init__()
        self.title("Monument AI - Elite Analytics (DnD Supported)")
        self.geometry("1200x850")

        # Load Model
        print(" Loading Model...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # UI State
        self.current_accent = THEMES["Ocean Blue"]["accent"]

        # --- GRID SYSTEM ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 📂 SIDEBAR (Glassmorphism) =================
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.logo = ctk.CTkLabel(
            self.sidebar, text="🏛️ MONUMENT AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo.pack(pady=(30, 5))
        self.sub_logo = ctk.CTkLabel(
            self.sidebar, text="V5 Elite Edition", font=ctk.CTkFont(size=12), text_color="gray")
        self.sub_logo.pack(pady=(0, 20))


        self.btn_load = ctk.CTkButton(
            self.sidebar, text="📁 Select Image", height=40, command=self.load_image_dialog)
        self.btn_load.pack(pady=10, padx=20)

        self.btn_clear = ctk.CTkButton(
            self.sidebar, text="🧹 Reset View", fg_color="transparent", border_width=2, command=self.clear_ui)
        self.btn_clear.pack(pady=10, padx=20)


        ctk.CTkLabel(self.sidebar, text="System Theme:", font=ctk.CTkFont(
            size=13, weight="bold")).pack(pady=(30, 5))
        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar, values=list(THEMES.keys()), command=self.change_theme)
        self.theme_menu.pack(pady=10, padx=20)

        self.hw_info = ctk.CTkLabel(self.sidebar, text="Hardware: RTX 3050 Active ✅", font=ctk.CTkFont(
            size=11), text_color="#2ecc71")
        self.hw_info.pack(side="bottom", pady=20)

        # ================= VIEWPORT (DnD Zone) =================
        self.main_content = ctk.CTkScrollableFrame(
            self, corner_radius=15, fg_color="#121212")
        self.main_content.grid(row=0, column=1, padx=20,
                               pady=20, sticky="nsew")

        self.main_content.drop_target_register(DND_FILES)
        self.main_content.dnd_bind('<<Drop>>', self.handle_drop)

        self.view_cards = []
        titles = ["Original (RGB)", "Depth Map", "Grayscale", "Edge Features"]
        for i in range(4):
            card = ctk.CTkFrame(self.main_content, corner_radius=15,
                                border_width=1, border_color="#333333")
            card.grid(row=i//2, column=i % 2, padx=15, pady=15)

            ctk.CTkLabel(card, text=titles[i], font=ctk.CTkFont(
                size=14, weight="bold")).pack(pady=5)
            img_container = ctk.CTkLabel(card, text="DRAP & DROP HERE", text_color="#444444",
                                         fg_color="black", width=380, height=380, corner_radius=10)
            img_container.pack(padx=10, pady=10)
            self.view_cards.append(img_container)

        # =================  RESULT BAR =================
        self.res_card = ctk.CTkFrame(self, height=120, corner_radius=15)
        self.res_card.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")

        self.res_label = ctk.CTkLabel(
            self.res_card, text="System Ready - Drag an image to begin", font=ctk.CTkFont(size=20, weight="bold"))
        self.res_label.pack(pady=(15, 5))

        self.conf_bar = ctk.CTkProgressBar(self.res_card, width=700, height=10)
        self.conf_bar.set(0)
        self.conf_bar.pack(pady=10)

    # --- LOGIC FUNCTIONS ---
    def handle_drop(self, event):
        path = event.data
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1] 
        if path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.process_image(path)

    def load_image_dialog(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.process_image(path)

    def process_image(self, path):
        self.res_label.configure(
            text="🧠 Analyzing Structure...", text_color="orange")
        self.update()

        views = get_views(path, augment=False)
        if views is None:
            return

        for i, view in enumerate(views):
            disp_img = (view * 255).astype(np.uint8)
            img_pil = Image.fromarray(disp_img).resize((380, 380))
            img_tk = ctk.CTkImage(img_pil, size=(380, 380))
            self.view_cards[i].configure(image=img_tk, text="")

        input_data = [np.expand_dims(v, axis=0) for v in views]
        preds = self.model.predict(input_data, verbose=0)
        idx = np.argmax(preds)
        conf = float(preds[0][idx])

        self.res_label.configure(
            text=f"🏛️  {CLASSES[idx].replace('_', ' ')}  ({conf*100:.2f}%)", text_color=self.current_accent)
        self.conf_bar.set(conf)
        self.conf_bar.configure(progress_color=self.current_accent)

    def change_theme(self, choice):
        palette = THEMES[choice]
        self.current_accent = palette["accent"]

        self.btn_load.configure(
            fg_color=palette["accent"], hover_color=palette["hover"])
        self.theme_menu.configure(
            button_color=palette["accent"], button_hover_color=palette["hover"])
        self.conf_bar.configure(progress_color=palette["accent"])
        self.res_label.configure(text_color=palette["accent"])

    def clear_ui(self):
        for card in self.view_cards:
            card.configure(image=None, text="DRAG & DROP HERE")
        self.res_label.configure(text="System Ready", text_color="white")
        self.conf_bar.set(0)


if __name__ == "__main__":
    app = MonumentAI_Elite_GUI()
    app.mainloop()
