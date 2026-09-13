# 🚗 Vehicle Color Detection

Detect vehicles in a video and classify each one's color — **Black, White, Red, Blue, or Silver** — using **YOLOv8** for detection and **OpenCV (HSV color thresholding)** for color analysis.

Built to be lightweight and beginner-friendly: a single, fully-commented Python script that runs on CPU, no GPU required.

---

## ✨ Features

- 🔍 Detects **cars, trucks, and buses** using YOLOv8 (`yolov8n.pt` — fast, CPU-friendly)
- 🎨 Classifies each vehicle's dominant color into 5 categories using HSV thresholding
- 🟩 Draws bounding boxes and labels (e.g. `Car: Blue`) directly on the video
- 💾 Reads `input_video.mp4` and writes an annotated `output_video.mp4`
- 🛡️ Safe video handling — files are always closed properly, even on error or interruption
- 📝 Every line explained with beginner-friendly comments

---

## 📦 Requirements

- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/)
- opencv-python
- numpy

Install everything with:

```bash
pip install ultralytics opencv-python numpy
```

---

## 🚀 Usage

1. Place your video in the project folder and name it `input_video.mp4`
2. Run the script:

```bash
python vehicle_color_detection.py
```

3. The first run will automatically download the `yolov8n.pt` model weights.
4. Once processing finishes, find your annotated video at `output_video.mp4`.

---

## ⚙️ How It Works

1. **Detection** — Each frame is passed through YOLOv8, which returns bounding boxes for cars, trucks, and buses (COCO class IDs `2`, `5`, `7`).
2. **Cropping** — Each detected vehicle is cropped out of the frame.
3. **Color Classification** — The crop is resized and center-cropped (to avoid windows, tires, and shadows), converted to HSV, and matched against threshold ranges:
   - **Red / Blue** — hue-based ranges
   - **Black / White / Silver** — saturation & brightness (value) based ranges
   
   The color covering the most pixels wins. If no color covers at least 5% of the crop, the result is labeled `Unknown` rather than guessing.
4. **Drawing** — A bounding box and `"VehicleType: Color"` label are drawn on the frame.
5. **Output** — Annotated frames are written to `output_video.mp4`.

---

## 🔧 Configuration

All key settings live at the top of the script:

| Setting | Description | Default |
|---|---|---|
| `INPUT_VIDEO_PATH` | Input video filename | `input_video.mp4` |
| `OUTPUT_VIDEO_PATH` | Output video filename | `output_video.mp4` |
| `YOLO_MODEL_PATH` | YOLOv8 model weights | `yolov8n.pt` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.4` |
| `MIN_COLOR_PIXEL_RATIO` | Minimum pixel coverage to trust a color | `0.05` (5%) |

---

## ⚠️ Limitations

- HSV threshold-based classification is approximate — lighting, reflections, and shadows can affect accuracy (especially silver vs. white, or dark blue vs. black).
- Only 5 color categories are supported; other colors (green, yellow, etc.) will be labeled `Unknown`.
- Only detects cars, trucks, and buses — motorcycles and other vehicles are ignored by default (easy to add via `VEHICLE_CLASS_IDS`).
- Runs on CPU by default via `yolov8n.pt`; larger videos will process more slowly than with a GPU.

---

## 🛠️ Possible Improvements

- Train/use a small CNN classifier for more robust color detection
- Add vehicle tracking (e.g. with `ByteTrack`) to avoid re-classifying the same vehicle every frame
- Support additional colors and vehicle types
- Add a summary report (counts per color/vehicle type) at the end of processing

---

## 📄 License

Free to use and modify for personal or commercial projects.
