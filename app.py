"""Desktop drawing pad for handwritten Chinese digit recognition."""

import argparse
from pathlib import Path
import tkinter as tk

import torch
from PIL import Image, ImageDraw, ImageOps

from chinese_digits.inference import predict_image
from chinese_digits.model import load_model


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the handwritten Chinese digit recognition demo."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pthmnist_cnn.pth",
        help="path to a compatible PyTorch state dictionary",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda",
    )
    return parser


class DrawingApp(tk.Tk):
    def __init__(self, model, device: str) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.title("Chinese Digit Recognition")
        self.geometry("420x540")
        self.resizable(False, False)
        self.configure(bg="#0f172a")

        tk.Label(
            self,
            text="Handwritten Chinese Digits",
            font=("Segoe UI", 20, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack(pady=(22, 4))
        tk.Label(
            self,
            text="Draw one character, then ask the CNN to identify it.",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(pady=(0, 16))

        self.canvas = tk.Canvas(
            self,
            bg="white",
            width=320,
            height=320,
            highlightthickness=3,
            highlightbackground="#334155",
        )
        self.canvas.pack()
        self.image = Image.new("L", (320, 320), color=255)
        self.drawer = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        button_frame = tk.Frame(self, bg="#0f172a")
        button_frame.pack(pady=16)
        tk.Button(
            button_frame,
            text="Recognize",
            command=self.recognize,
            width=15,
            bg="#22c55e",
            fg="#052e16",
            activebackground="#16a34a",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=6)
        tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            width=11,
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=6)

        self.result_label = tk.Label(
            self,
            text="Ready",
            font=("Microsoft YaHei UI", 17, "bold"),
            fg="#e2e8f0",
            bg="#0f172a",
        )
        self.result_label.pack()

    def paint(self, event) -> None:
        radius = 11
        bounds = (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        )
        self.canvas.create_oval(*bounds, fill="black", outline="black")
        self.drawer.ellipse(bounds, fill=0)

    def clear(self) -> None:
        self.canvas.delete("all")
        self.drawer.rectangle((0, 0, 320, 320), fill=255)
        self.result_label.configure(text="Ready", fg="#e2e8f0")

    def recognize(self) -> None:
        if self.image.getbbox() is None or ImageOps.invert(self.image).getbbox() is None:
            self.result_label.configure(text="Draw a character first", fg="#fbbf24")
            return
        model_input = ImageOps.invert(self.image)
        prediction = predict_image(self.model, model_input, self.device)
        self.result_label.configure(
            text=(
                f"{prediction.character}  ·  {prediction.value:,}  "
                f"({prediction.confidence:.1%})"
            ),
            fg="#86efac",
        )


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.weights, device=args.device)
    DrawingApp(model, args.device).mainloop()


if __name__ == "__main__":
    main()
