import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import threading


# =========================
# Model Architectures
# =========================

class BaselineCNN(nn.Module):
    """
    Baseline multi-task CNN trained from scratch.
    """
    def __init__(self, num_fruits=3, dropout_p=0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1 => Handle stuff like Edges
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2 => Handle stuff like Textures
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3 => Handle stuff like Patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global Average Pool
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Shared Feature Vector (fruit)
        self.fruit_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_fruits)
        )
        
        # Shared Feature Vector (ripeness): outputs a score in [0, 1]
        self.ripeness_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        fruit_logits = self.fruit_head(x)
        ripeness = self.ripeness_head(x).squeeze(1)
        return fruit_logits, ripeness


class AdvancedCNN(nn.Module):
    """
    Multi-task CNN using a pretrained ResNet18 backbone.
    """
    def __init__(self, num_fruits=3):
        super().__init__()

        # Pretrained ResNet18 backbone
        backbone = models.resnet18(weights=None)  # We'll load weights from file
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        feat_dim = 512  # ResNet18 feature dimension

        # Fruit classification head
        self.fruit_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_fruits)
        )

        # Ripeness regression head
        self.ripeness_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature extraction
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)

        fruit_logits = self.fruit_head(feats)
        ripeness = self.ripeness_head(feats).squeeze(0)

        return fruit_logits, ripeness


# =========================
# Prediction Helper Functions
# =========================

FRUITS = ["apple", "banana", "orange"]
STAGES = ["unripe", "fresh", "rotten"]
STAGE_SCORES = {"unripe": 0.0, "fresh": 0.5, "rotten": 1.0}

# Image preprocessing transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


def score_to_stage(score):
    """Convert continuous ripeness score to discrete stage."""
    if score < 0.25:
        return "unripe"
    elif score < 0.75:
        return "fresh"
    else:
        return "rotten"


def predict_image(model, image_path, device):
    """
    Make prediction on a single image.
    
    Returns:
        tuple: (fruit_name, stage_name, confidence, ripeness_score)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        fruit_logits, ripeness_score = model(image_tensor)
        
        # Get fruit prediction
        fruit_probs = torch.softmax(fruit_logits, dim=1)
        fruit_confidence, fruit_idx = torch.max(fruit_probs, dim=1)
        fruit_name = FRUITS[fruit_idx.item()]
        
        # Get stage from ripeness score
        ripeness_value = ripeness_score.item()
        stage_name = score_to_stage(ripeness_value)
        
    return fruit_name, stage_name, fruit_confidence.item(), ripeness_value


# =========================
# Main Application
# =========================

class CnnDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        # Use a nicer ttk theme
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Basic window setup
        self.root.title("Fruit Ripeness Detector")
        self.root.geometry("700x750")
        self.root.minsize(700, 750)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model setup
        self.models = {}
        self.current_model_name = "advanced"  # Start with advanced model
        self.load_models()

        # Keep a reference to the displayed image
        self.photo_ref = None
        self.selected_image_path = None

        # Store canvas ids for the clear (X) button
        self.clear_btn_id = None
        self.clear_box_id = None

        # Track if processing
        self.is_processing = False

        # Build UI
        self.build_ui()

    def load_models(self):
        """Load both trained models."""
        try:
            # Load baseline model
            print("Loading baseline model...")
            baseline_model = BaselineCNN(num_fruits=3)
            baseline_model.load_state_dict(
                torch.load("baseline_model.pth", map_location=self.device)
            )
            baseline_model.to(self.device)
            baseline_model.eval()
            self.models["baseline"] = baseline_model
            print("Baseline model loaded")

            # Load advanced model
            print("Loading advanced model...")
            advanced_model = AdvancedCNN(num_fruits=3)
            advanced_model.load_state_dict(
                torch.load("advanced_model.pth", map_location=self.device)
            )
            advanced_model.to(self.device)
            advanced_model.eval()
            self.models["advanced"] = advanced_model
            print("Advanced model loaded")

        except FileNotFoundError as e:
            messagebox.showerror(
                "Model Error",
                f"Could not find model files!\n\n"
                f"Please ensure 'baseline_model.pth' and 'advanced_model.pth' "
                f"are in the same directory as this script.\n\n"
                f"Error: {e}"
            )
            self.root.quit()
        except Exception as e:
            messagebox.showerror(
                "Model Error",
                f"Error loading models:\n{e}"
            )
            self.root.quit()

    def build_ui(self):
        """Build the user interface."""
        # Main container frame
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        # Title
        title = ttk.Label(
            container,
            text="Fruit Ripeness Detector",
            font=("Helvetica", 20, "bold"),
            anchor="center",
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Model selector
        model_frame = ttk.Frame(container)
        model_frame.grid(row=1, column=0, pady=(0, 10))
        
        ttk.Label(
            model_frame,
            text="Model:",
            font=("Helvetica", 10)
        ).pack(side="left", padx=(0, 10))
        
        self.model_var = tk.StringVar(value="baseline")

        baseline_radio = ttk.Radiobutton(
            model_frame,
            text="Baseline (Custom CNN)",
            variable=self.model_var,
            value="baseline",
            command=self.on_model_change
        )
        baseline_radio.pack(side="left", padx=5)
        
        advanced_radio = ttk.Radiobutton(
            model_frame,
            text="Advanced (ResNet18)",
            variable=self.model_var,
            value="advanced",
            command=self.on_model_change
        )
        advanced_radio.pack(side="left", padx=5)


        # Result subtitle
        self.result_var = tk.StringVar(value="")
        result_label = ttk.Label(
            container,
            textvariable=self.result_var,
            font=("Helvetica", 13, "bold"),
            anchor="center",
        )
        result_label.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        # Confidence/details label
        self.details_var = tk.StringVar(value="")
        details_label = ttk.Label(
            container,
            textvariable=self.details_var,
            font=("Helvetica", 10),
            anchor="center",
        )
        details_label.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        # Image preview area
        self.preview_w = 500
        self.preview_h = 500

        self.canvas = tk.Canvas(
            container,
            width=self.preview_w,
            height=self.preview_h,
            bg="#1f1f1f",
            highlightthickness=2,
            highlightbackground="#444",
        )
        self.canvas.grid(row=4, column=0, pady=(0, 15))

        # Show placeholder initially
        self.draw_placeholder()

        # Buttons row
        btn_row = ttk.Frame(container)
        btn_row.grid(row=5, column=0)

        select_btn = ttk.Button(
            btn_row, 
            text="Select Image", 
            command=self.select_image
        )
        select_btn.pack(side="left")

        self.analyse_btn = ttk.Button(
            btn_row,
            text="Analyse",
            command=self.process_image,
            state="disabled"
        )
        self.analyse_btn.pack(side="left", padx=10)

        # Handle canvas clicks
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_model_change(self):
        """Handle model selection change."""
        self.current_model_name = self.model_var.get()
        # Clear results when switching models
        if not self.is_processing:
            self.result_var.set("")
            self.details_var.set("")

    def draw_placeholder(self):
        """Clear canvas and show default placeholder message."""
        self.canvas.delete("all")

        self.canvas.create_text(
            self.preview_w // 2,
            self.preview_h // 2,
            text="No image selected",
            fill="#cccccc",
            font=("Helvetica", 18, "bold"),
        )
        self.canvas.create_text(
            self.preview_w // 2,
            self.preview_h // 2 + 30,
            text="Click 'Select Image' to upload",
            fill="#aaaaaa",
            font=("Helvetica", 12),
        )

        self.clear_btn_id = None
        self.clear_box_id = None

    def draw_clear_button(self):
        """Draw a small 'X' button at top-right of canvas."""
        x1, y1 = self.preview_w - 35, 10
        x2, y2 = self.preview_w - 10, 35

        self.clear_box_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="#2b2b2b",
            outline="#555"
        )

        self.clear_btn_id = self.canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text="✕",
            fill="#dddddd",
            font=("Helvetica", 14, "bold"),
        )

    def hide_clear_button(self):
        """Remove the clear button from canvas."""
        if self.clear_box_id is not None:
            self.canvas.delete(self.clear_box_id)
            self.clear_box_id = None

        if self.clear_btn_id is not None:
            self.canvas.delete(self.clear_btn_id)
            self.clear_btn_id = None

    def select_image(self):
        """Open file picker and load selected image."""
        if self.is_processing:
            return

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self.selected_image_path = file_path
        self.analyse_btn.config(state="normal")
        self.result_var.set("")
        self.details_var.set("")

        # Display image
        self.canvas.delete("all")

        img = Image.open(file_path).convert("RGB")
        img.thumbnail((self.preview_w, self.preview_h))

        self.photo_ref = ImageTk.PhotoImage(img)

        self.canvas.create_image(
            self.preview_w // 2,
            self.preview_h // 2,
            image=self.photo_ref,
            anchor="center",
        )

        self.draw_clear_button()

    def clear_image(self):
        """Clear selected image and reset UI."""
        if self.is_processing:
            return

        self.selected_image_path = None
        self.photo_ref = None
        self.analyse_btn.config(state="disabled")
        self.result_var.set("")
        self.details_var.set("")
        self.draw_placeholder()

    def on_canvas_click(self, event):
        """Handle clicks on canvas (for clear button)."""
        if self.is_processing:
            return

        if self.clear_btn_id is None or self.clear_box_id is None:
            return

        clicked = self.canvas.find_closest(event.x, event.y)
        if not clicked:
            return

        if clicked[0] in (self.clear_btn_id, self.clear_box_id):
            self.clear_image()

    def process_image(self):
        """Process the selected image with the current model."""
        if not self.selected_image_path or self.is_processing:
            return

        self.is_processing = True
        self.analyse_btn.config(state="disabled")
        self.hide_clear_button()

        # Show loading
        self.result_var.set("Analysing...")
        self.details_var.set("Please wait...")

        # Run prediction in separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_prediction)
        thread.daemon = True
        thread.start()

    def run_prediction(self):
        """Run the actual prediction (called in separate thread)."""
        try:
            model = self.models[self.current_model_name]
            
            fruit, stage, confidence, ripeness = predict_image(
                model, 
                self.selected_image_path, 
                self.device
            )

            # Update UI in main thread
            self.root.after(0, self.show_results, fruit, stage, confidence, ripeness)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def show_results(self, fruit, stage, confidence, ripeness):
        """Display prediction results."""
        # Main result
        result_text = f"{fruit.capitalize()} - {stage.capitalize()}"
        self.result_var.set(result_text)

        # Details
        model_display = "Advanced Model" if self.current_model_name == "advanced" else "Baseline Model"
        details_text = (
            f"Model: {model_display} | "
            f"Confidence: {confidence*100:.1f}% | "
            f"Ripeness Score: {ripeness:.2f}"
        )
        self.details_var.set(details_text)

        # Re-enable button and show clear button
        self.is_processing = False
        if self.selected_image_path:
            self.analyse_btn.config(state="normal")
            self.draw_clear_button()

    def show_error(self, error_msg):
        """Display error message."""
        self.result_var.set("Error during analysis")
        self.details_var.set(f"Error: {error_msg}")
        
        self.is_processing = False
        if self.selected_image_path:
            self.analyse_btn.config(state="normal")
            self.draw_clear_button()


def main():
    print("Starting Fruit Ripeness Detector UI...")
    root = tk.Tk()
    app = CnnDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
