"""
================================================================================
 VEHICLE COLOR DETECTION
================================================================================
What this script does (in plain English):

1.  It opens a video file called 'input_video.mp4'.
2.  For every single frame of that video, it uses a pre-trained AI model
    called YOLOv8 (You Only Look Once, version 8) to find vehicles
    (cars, trucks, and buses) in the frame.
3.  For every vehicle it finds, it "crops" (cuts out) just that vehicle
    from the frame as a small image.
4.  It then analyzes the colors inside that small cropped image using
    OpenCV's HSV color space to guess the vehicle's color. It can guess
    5 colors: Black, White, Red, Blue, or Silver.
5.  It draws a rectangle (bounding box) around the vehicle on the frame,
    and writes a text label above it, like "Car: Blue".
6.  Finally, it saves every processed frame into a brand new video file
    called 'output_video.mp4'.

Requirements (install these first if you haven't already):
    pip install ultralytics opencv-python numpy

Author note: This script is heavily commented so that even someone who
has never touched computer vision code before can follow along.
================================================================================
"""

# ------------------------------------------------------------------------
# STEP 0: IMPORT THE LIBRARIES WE NEED
# ------------------------------------------------------------------------
import os                      # Used to check whether files exist on disk
import sys                     # Used to cleanly stop the program if something is wrong
import cv2                     # OpenCV -> used for video reading/writing and color analysis
import numpy as np             # NumPy -> used for fast array/number operations
from ultralytics import YOLO   # The YOLOv8 object detection model


# ------------------------------------------------------------------------
# STEP 1: CONFIGURATION (all the "settings" for this script live here)
# ------------------------------------------------------------------------

# The name of the video file we want to read (must be in the same folder
# as this script, unless you give a full path).
INPUT_VIDEO_PATH = "input_video.mp4"

# The name of the video file we will create with our results drawn on it.
OUTPUT_VIDEO_PATH = "output_video.mp4"

# Which YOLOv8 model file to use. "yolov8n.pt" is the "nano" version,
# which is the smallest and fastest YOLOv8 model, ideal for running on a
# normal CPU (no expensive graphics card needed). Ultralytics will
# automatically download this file the first time you run the script
# if it isn't already present on your computer.
YOLO_MODEL_PATH = "yolov8n.pt"

# Minimum confidence score (0.0 to 1.0) YOLO must have before we trust a
# detection. Raising this reduces false detections; lowering it detects
# more (but possibly wrong) objects.
CONFIDENCE_THRESHOLD = 0.4

# The YOLO model is trained on the COCO dataset, where every object type
# ("class") has a fixed ID number. The ones we care about for vehicles are:
#   2 = car
#   5 = bus
#   7 = truck
# (Class 3, motorcycle, is intentionally left out since it wasn't requested.)
VEHICLE_CLASS_IDS = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}

# When we crop out a vehicle and look at its colors, some pixels might be
# "noise" (shadows, reflections, tiny background slivers, etc). We only
# trust a color as the final answer if it covers at least this fraction
# of the cropped image. Otherwise, we label it "Unknown".
MIN_COLOR_PIXEL_RATIO = 0.05  # 5%


# ------------------------------------------------------------------------
# STEP 2: DEFINE THE COLOR RANGES (in HSV) FOR EACH COLOR WE CARE ABOUT
# ------------------------------------------------------------------------
#
# Why HSV and not plain RGB/BGR?
# -------------------------------
# In HSV (Hue, Saturation, Value):
#   - Hue        tells us the actual "color" (red, blue, green, etc.)
#   - Saturation tells us how "pure"/vivid the color is (0 = gray, 255 = vivid)
#   - Value      tells us how bright/dark the color is (0 = black, 255 = bright)
#
# This separation makes it MUCH easier to detect things like "black",
# "white", and "silver/gray" because those are not really "hues" at all —
# they are defined by low saturation and/or specific brightness levels,
# no matter what the hue is. Regular BGR color values don't separate
# brightness from color, so thresholding in BGR is much harder.
#
# NOTE: OpenCV represents Hue on a scale of 0-179 (not 0-360) because it
# fits into a single 8-bit channel. Saturation and Value both range 0-255.

# Colors that ARE defined mainly by hue (red and blue).
# Each entry is a list of (lower_bound, upper_bound) HSV tuples.
# Red needs TWO ranges because red wraps around the hue circle (it sits
# at both the very start (0) and the very end (179) of the hue scale).
HUE_BASED_COLOR_RANGES = {
    "Red": [
        (np.array([0,   70, 50]),  np.array([10,  255, 255])),   # low end of hue
        (np.array([170, 70, 50]),  np.array([180, 255, 255])),   # high end of hue
    ],
    "Blue": [
        (np.array([94, 80, 2]),    np.array([126, 255, 255])),
    ],
}

# Colors that are defined mainly by Saturation (S) and Value/brightness (V),
# regardless of hue. Hue is left wide open (0-180) for all three.
BRIGHTNESS_BASED_COLOR_RANGES = {
    # Black: very dark, regardless of how saturated the color looks.
    "Black": (np.array([0, 0, 0]),      np.array([180, 255, 50])),

    # White: bright AND very low saturation (i.e. not really "colorful").
    "White": (np.array([0, 0, 200]),    np.array([180, 30, 255])),

    # Silver / Gray: mid-brightness and low saturation. This sits between
    # black and white in terms of brightness.
    "Silver": (np.array([0, 0, 50]),    np.array([180, 40, 200])),
}


# ------------------------------------------------------------------------
# STEP 3: THE COLOR DETECTION FUNCTION
# ------------------------------------------------------------------------
def detect_vehicle_color(vehicle_crop_bgr):
    """
    Given a cropped image of a single vehicle (in standard OpenCV BGR
    format), this function figures out which of the 5 known colors
    (Black, White, Red, Blue, Silver) covers the LARGEST portion of the
    vehicle's surface, and returns that color's name as a string.

    If no color range covers enough of the image, it returns "Unknown".
    """

    # Safety check: if the crop is empty (e.g., the detection box was at
    # the very edge of the frame and got cropped to 0 pixels), we can't
    # analyze anything, so just bail out early.
    if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
        return "Unknown"

    # Resize the crop to a small, fixed size. This makes the analysis
    # much faster (fewer pixels to check) and also reduces the influence
    # of tiny background slivers or noise around the edges of the box.
    resized_crop = cv2.resize(vehicle_crop_bgr, (60, 60), interpolation=cv2.INTER_AREA)

    # To focus on the vehicle's body paint and avoid windows, tires, and
    # shadows, we only analyze the CENTER portion of the resized crop.
    # This is a simple trick: cars usually have their main paint color
    # dominating the middle of the bounding box.
    h, w = resized_crop.shape[:2]
    margin_h, margin_w = int(h * 0.25), int(w * 0.25)
    center_region = resized_crop[margin_h:h - margin_h, margin_w:w - margin_w]

    # Convert the cropped region from BGR (OpenCV's default) to HSV,
    # since our color ranges above were defined in HSV.
    hsv_image = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

    # Total number of pixels we're analyzing -> used later to calculate
    # what PERCENTAGE of the image each color covers.
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    if total_pixels == 0:
        return "Unknown"

    # This dictionary will store: { "ColorName": pixel_count_matching_that_color }
    color_pixel_counts = {}

    # --- Check the hue-based colors (Red, Blue) ---
    for color_name, ranges in HUE_BASED_COLOR_RANGES.items():
        # Start with an all-zero (all black) mask the same size as our image.
        combined_mask = np.zeros((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8)

        # A color might need multiple ranges (like Red), so loop through
        # all of them and combine ("OR") the masks together.
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # cv2.countNonZero counts how many pixels are "white" (matched) in the mask.
        color_pixel_counts[color_name] = cv2.countNonZero(combined_mask)

    # --- Check the brightness-based colors (Black, White, Silver) ---
    for color_name, (lower, upper) in BRIGHTNESS_BASED_COLOR_RANGES.items():
        mask = cv2.inRange(hsv_image, lower, upper)
        color_pixel_counts[color_name] = cv2.countNonZero(mask)

    # Find whichever color had the highest matching pixel count.
    best_color_name = max(color_pixel_counts, key=color_pixel_counts.get)
    best_color_pixel_count = color_pixel_counts[best_color_name]

    # Calculate what fraction of the total image that best color covers.
    coverage_ratio = best_color_pixel_count / float(total_pixels)

    # If even the "best" color doesn't cover enough of the image, we
    # don't trust the result, so we label it as "Unknown" instead of
    # guessing wrong.
    if coverage_ratio < MIN_COLOR_PIXEL_RATIO:
        return "Unknown"

    return best_color_name


# ------------------------------------------------------------------------
# STEP 4: HELPER FUNCTION TO DRAW THE BOUNDING BOX + LABEL ON THE FRAME
# ------------------------------------------------------------------------
def draw_detection(frame, box_coords, label_text):
    """
    Draws a green rectangle around the detected vehicle, plus a filled
    label background and white text showing the vehicle type and color.
    """
    x1, y1, x2, y2 = box_coords

    box_color = (0, 255, 0)          # Green, in BGR format
    text_color = (255, 255, 255)     # White text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Draw the main bounding box rectangle around the vehicle.
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    # Figure out how big the label text will be, so we can draw a
    # nicely-fitted background rectangle behind it (makes text readable
    # even over busy/bright backgrounds).
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )

    # Position the label just above the top-left corner of the box.
    # If the box is too close to the top of the frame, place the label
    # just BELOW the top edge instead, so it doesn't get cut off.
    label_bg_y1 = max(0, y1 - text_height - baseline - 4)
    label_bg_y2 = y1 if y1 - text_height - baseline - 4 >= 0 else y1 + text_height + baseline + 4
    label_text_y = label_bg_y2 - baseline - 2

    cv2.rectangle(
        frame,
        (x1, label_bg_y1),
        (x1 + text_width + 4, label_bg_y2),
        box_color,
        thickness=-1,  # -1 means "filled" rectangle
    )
    cv2.putText(
        frame, label_text, (x1 + 2, label_text_y),
        font, font_scale, text_color, thickness, lineType=cv2.LINE_AA
    )


# ------------------------------------------------------------------------
# STEP 5: THE MAIN PROGRAM
# ------------------------------------------------------------------------
def main():
    # --- 5a. Make sure the input video actually exists before we do anything else ---
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"ERROR: Could not find '{INPUT_VIDEO_PATH}'. "
              f"Please place your input video in the same folder as this "
              f"script and make sure it's named exactly '{INPUT_VIDEO_PATH}'.")
        sys.exit(1)

    # --- 5b. Load the YOLOv8 model ---
    # The first time this runs, Ultralytics will automatically download
    # the 'yolov8n.pt' weights file from the internet if it's not already
    # saved locally. This might take a few seconds.
    print(f"Loading YOLO model '{YOLO_MODEL_PATH}' ... (this may take a moment)")
    model = YOLO(YOLO_MODEL_PATH)
    print("Model loaded successfully!")

    # --- 5c. Open the input video for reading ---
    video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not video_capture.isOpened():
        print(f"ERROR: Could not open video file '{INPUT_VIDEO_PATH}'. "
              f"It might be corrupted or in an unsupported format.")
        sys.exit(1)

    # Grab some basic info about the video so our output video matches it:
    #   - width/height of each frame (in pixels)
    #   - frames per second (fps)
    #   - total frame count (just used to show progress to the user)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Safety fallback: some video files report fps as 0, which would break
    # the output video writer. Default to 25 fps if that happens.
    if fps is None or fps <= 0:
        fps = 25.0

    print(f"Input video info -> width: {frame_width}, height: {frame_height}, "
          f"fps: {fps:.2f}, total frames: {total_frames}")

    # --- 5d. Set up the video writer for our output video ---
    # 'mp4v' is a widely-compatible codec for writing .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        print(f"ERROR: Could not create output video file '{OUTPUT_VIDEO_PATH}'.")
        video_capture.release()
        sys.exit(1)

    # We wrap the main processing loop in a try/except/finally block.
    # This is a SAFEGUARD: no matter what happens (the code finishes
    # normally, the user presses Ctrl+C, or an unexpected error occurs),
    # the 'finally' block at the bottom will ALWAYS run and properly
    # close/release the video files. This prevents corrupted or
    # unplayable output videos.
    frame_number = 0
    try:
        print("Starting video processing... Press Ctrl+C to stop early.")

        # This loop runs once per frame of the video, until there are no
        # more frames left to read.
        while True:
            # video_capture.read() gives us two things:
            #   success -> True if a frame was successfully read, False if
            #              we've reached the end of the video (or an error).
            #   frame   -> the actual image (as a NumPy array) for this frame.
            success, frame = video_capture.read()

            if not success:
                # No more frames left -> we've reached the end of the video.
                break

            frame_number += 1

            # ------------------------------------------------------------
            # Run YOLO object detection on this single frame.
            # 'verbose=False' just stops YOLO from printing a log line
            # for every single frame (keeps our console output clean).
            # ------------------------------------------------------------
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # YOLO returns a list of "Results" objects (one per image we
            # gave it). Since we only gave it one frame, we only care
            # about results[0].
            detections = results[0].boxes

            # Loop through every object YOLO detected in this frame.
            for box in detections:
                # class_id tells us WHAT kind of object this is (car, dog,
                # person, etc.) according to the COCO dataset's numbering.
                class_id = int(box.cls[0])

                # We only care about vehicles (car, bus, truck). Skip
                # anything else (people, animals, traffic lights, etc.)
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                vehicle_type_name = VEHICLE_CLASS_IDS[class_id]

                # box.xyxy gives us the bounding box coordinates in the
                # format [x1, y1, x2, y2], where (x1,y1) is the top-left
                # corner and (x2,y2) is the bottom-right corner, measured
                # in pixels from the top-left of the frame.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Safety clamp: make sure our coordinates never go outside
                # the actual frame boundaries (this can occasionally
                # happen due to rounding).
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y2 = max(0, min(y2, frame_height - 1))

                # Skip any boxes that ended up with zero width or height
                # after clamping (degenerate/invalid boxes).
                if x2 <= x1 or y2 <= y1:
                    continue

                # ------------------------------------------------------------
                # Crop the vehicle out of the frame using array slicing.
                # NumPy/OpenCV images are indexed as [row(y), column(x)],
                # so height comes before width here.
                # ------------------------------------------------------------
                vehicle_crop = frame[y1:y2, x1:x2]

                # Figure out the vehicle's dominant color using our
                # color-detection function from Step 3.
                vehicle_color = detect_vehicle_color(vehicle_crop)

                # Build the label text, e.g. "Car: Blue"
                label_text = f"{vehicle_type_name}: {vehicle_color}"

                # Draw the bounding box + label directly onto the frame.
                draw_detection(frame, (x1, y1, x2, y2), label_text)

            # Write this now-annotated frame into our output video file.
            video_writer.write(frame)

            # Print progress every 30 frames so the user knows it's working.
            if frame_number % 30 == 0:
                if total_frames > 0:
                    percent_done = (frame_number / total_frames) * 100
                    print(f"Processed frame {frame_number}/{total_frames} "
                          f"({percent_done:.1f}%)")
                else:
                    print(f"Processed frame {frame_number}...")

        print(f"Finished! Processed {frame_number} frames in total.")

    except KeyboardInterrupt:
        # This runs if the user presses Ctrl+C to manually stop the script.
        print("\nProcessing interrupted by user (Ctrl+C). "
              "Saving what has been processed so far...")

    except Exception as error:
        # This is a general safety net for any unexpected error, so that
        # we still close our video files properly instead of leaving them
        # corrupted, and so the user sees a helpful message.
        print(f"\nAn unexpected error occurred: {error}")

    finally:
        # ------------------------------------------------------------
        # THIS BLOCK ALWAYS RUNS -> whether the script finished
        # normally, was interrupted, or crashed with an error.
        # Properly releasing (closing) both the video reader and writer
        # is essential -- otherwise the output .mp4 file can end up
        # corrupted or unplayable.
        # ------------------------------------------------------------
        video_capture.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"Video files closed safely. Output saved to '{OUTPUT_VIDEO_PATH}'.")


# ------------------------------------------------------------------------
# STEP 6: STANDARD PYTHON ENTRY POINT
# ------------------------------------------------------------------------
# This line means: "only run the main() function if this file is being
# run directly (e.g., `python vehicle_color_detection.py`), and NOT if
# it's being imported as a module inside another script."
if __name__ == "__main__":
    main()

