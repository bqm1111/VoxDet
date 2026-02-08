import os, re, cv2, numpy as np
from tqdm import tqdm
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def images_to_video(input_dir, output_path, fps=30.0,
                    exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff"),
                    codec="mp4v", resize_to=None):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    first_path = os.path.join(input_dir, files[0])
    first = cv2.imread(first_path, cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read {first_path}")

    if resize_to is None:
        h, w = first.shape[:2]
        frame_size = (w, h)
    else:
        frame_size = (int(resize_to[0]), int(resize_to[1]))

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), frame_size, True)
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed. Try a different codec/container. "
            f"codec={codec}, output={output_path}"
        )

    for name in tqdm(files):
        p = os.path.join(input_dir, name)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot read {p}")

        # Normalize to 8-bit BGR (VideoWriter expects 8-bit 3-channel for color=True)
        if img.dtype != np.uint8:
            # Handle 16-bit PNG/TIFF etc.
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img.ndim == 2:  # grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            raise RuntimeError(f"Unsupported channel count {img.shape} for {p}")

        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)

        ok = writer.write(img)
        # Some OpenCV builds return None; rely on warning + explicit sanity checks above.

    writer.release()

if __name__ == "__main__":
    # Use a safer codec/container combo if mp4v fails on your build:
    # - Try output.avi with XVID
    # - Or output.mp4 with avc1 (if your ffmpeg has H.264)
    images_to_video(
        input_dir="vis_compare/P008",
        output_path="output.avi",
        fps=5,
        codec="XVID",
        resize_to=None
    )
    