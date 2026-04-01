import sys
import os
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from contextlib import contextmanager


def log_status(message, verbose=False):
    if verbose:
        print(message, flush=True)


# Context manager to suppress stdout and stderr
@contextmanager
def silent_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Add directories to sys.path to allow imports
CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(CURRENT_DIR / 'MesoNet'))
sys.path.append(str(CURRENT_DIR / 'temp-d3'))

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL only

# Attempt imports with error handling to guide user on missing deps
# Wrap imports in silent_output to hide warnings
with silent_output():
    try:
        # MesoNet imports: direct import because we appended MesoNet dir
        from classifiers import Meso4
        from pipeline import FaceFinder, FaceBatchGenerator
        import tensorflow as tf # Verify TF presence
    except ImportError:
        # We can't print here if silent, but we re-raise or handle outside?
        # If import fails, we want to know. 
        # But commonly these just work if installed. 
        # Let's catch inside and toggle flag?
        pass

# We need to re-import safely or handle the scope. 
# Actually, imports inside context manager persist.
# But if it crashes, we won't see the error.
# Let's do a tailored approach: ONLY suppress unrelated warnings if possible, OR just suppress everything and if it fails, unexpected error.
# The user complains about warnings.
# Let's try to import normally but filter warnings? 
# No, `silent_output` is robust for "FutureWarning" etc.

try:
    with silent_output():
        from classifiers import Meso4
        from pipeline import FaceFinder, FaceBatchGenerator
        import tensorflow as tf
except ImportError as e:
    print(f"Error importing MesoNet modules: {e}")
    sys.exit(1)

try:
    with silent_output():
        from models.D3_model import D3_model
        import albumentations
        from data.datasets import crop_center_by_percentage, set_preprocessing
except ImportError as e:
    print(f"Error importing Temp-D3 modules: {e}")
    sys.exit(1)


def run_mesonet(video_path, weights_path, raise_on_error=False, verbose=False):
    """
    Runs MesoNet (Meso4) on the video.
    Returns a score (0.0 to 1.0), where higher means more likely to be Fake (deepfake).
    """
    if not os.path.exists(weights_path):
        # We want to see this error
        print(f"Error: MesoNet weights not found at {weights_path}")
        return None

    log_status("[MesoNet] Loading model weights...", verbose)
    # Load model
    with silent_output():
        classifier = Meso4()
        classifier.load(weights_path)
    log_status("[MesoNet] Model loaded.", verbose)

    # Face extraction
    try:
        log_status("[MesoNet] Extracting faces from video...", verbose)
        with silent_output():
            face_finder = FaceFinder(video_path, load_first_face=False)
        if face_finder.length == 0:
            log_status("[MesoNet] Video has 0 frames. Returning 0.0.", verbose)
            return 0.0

        # Keep behavior same as current implementation.
        skipstep = max(int(face_finder.length // 15), 1)
        log_status(
            f"[MesoNet] Frames detected: {face_finder.length}. Running face detection with skipstep={skipstep}...",
            verbose,
        )
        with silent_output():
            face_finder.find_faces(resize=0.5, skipstep=skipstep)

        face_count = len(face_finder.coordinates)
        if face_count == 0:
            log_status("[MesoNet] No faces found.", verbose)
            return None

        log_status(f"[MesoNet] Faces tracked on {face_count} frame(s). Running classifier...", verbose)
        gen = FaceBatchGenerator(face_finder)
        predictions = []
        batch_size = 50
        batch_index = 0

        batch = gen.next_batch(batch_size=batch_size)
        while len(batch) > 0:
            with silent_output():
                pred = classifier.predict(batch)
            predictions.extend(pred.flatten().tolist())
            batch_index += 1
            log_status(
                f"[MesoNet] Inference batch {batch_index} complete ({len(predictions)} predictions accumulated).",
                verbose,
            )
            if gen.head >= gen.length:
                break
            batch = gen.next_batch(batch_size=batch_size)

        if not predictions:
            log_status("[MesoNet] No predictions produced.", verbose)
            return None

        score = np.mean(predictions)
        log_status("[MesoNet] Score computed.", verbose)
        return score

    except Exception as e:
        if raise_on_error:
            raise RuntimeError(f"MesoNet inference failed: {e}") from e
        return None


def run_d3(video_path, encoder_type='XCLIP-16', loss_type='l2', raise_on_error=False, sample_across_video=False, verbose=False):
    """
    Runs Temp-D3 model on the video.
    Returns a score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_status(f"[Temp-D3] Loading model on device: {device}.", verbose)
    
    # Load Model
    try:
        with silent_output():
            model = D3_model(encoder_type=encoder_type, loss_type=loss_type)
            model.to(device)
            model.eval()
        log_status("[Temp-D3] Model loaded.", verbose)
    except Exception as e:
        if raise_on_error:
            raise RuntimeError(f"Temp-D3 model load failed: {e}") from e
        return None

    # Preprocessing
    try:
        log_status("[Temp-D3] Decoding and preprocessing video frames...", verbose)
        with silent_output():
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            log_status("[Temp-D3] Video has 0 frames. Returning 0.0.", verbose)
            return 0.0

        frames_to_read = min(total_frames, 16)
        if sample_across_video and total_frames > frames_to_read:
            frame_indices = np.linspace(0, total_frames - 1, frames_to_read, dtype=int).tolist()
        else:
            frame_indices = list(range(frames_to_read))

        log_status(
            f"[Temp-D3] Total frames: {total_frames}. Sampling {len(frame_indices)} frame(s).",
            verbose,
        )
        with silent_output():
            trans = set_preprocessing(None, None)

        collected_frames = []
        for frame_index in frame_indices:
            if sample_across_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop_center_by_percentage(frame, 0.1)
            augmented = trans(image=frame)
            frame_aug = augmented["image"]
            frame_tensor = frame_aug.transpose(2, 0, 1)
            collected_frames.append(frame_tensor)

        cap.release()

        if len(collected_frames) == 0:
            log_status("[Temp-D3] No valid frames decoded. Returning 0.0.", verbose)
            return 0.0

        input_tensor = np.stack(collected_frames, axis=0)
        input_tensor = input_tensor[np.newaxis, :]
        input_tensor = torch.tensor(input_tensor).float().to(device)

        log_status("[Temp-D3] Running inference...", verbose)
        with torch.no_grad():
            _, _, dis_std = model(input_tensor)
            score = dis_std.item()

        log_status("[Temp-D3] Score computed.", verbose)
        return score

    except Exception as e:
        if raise_on_error:
            raise RuntimeError(f"Temp-D3 inference failed: {e}") from e
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Deepfake Detection Models (MesoNet & Temp-D3)")
    parser.add_argument('video_path', type=str, help="Path to the video file to analyze")
    parser.add_argument('--verbose-status', action='store_true', help="Print stage-by-stage progress logs")
    args = parser.parse_args()
    
    video_path = args.video_path
    
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return

    # print(f"Analyzing video: {video_path}")
    # print("-" * 30)

    # --- Run MesoNet ---
    # print("Running MesoNet (Meso4)...")
    meso_weights = str(CURRENT_DIR / 'MesoNet/weights/Meso4_DF.h5')
    
    meso_score = run_mesonet(video_path, meso_weights, verbose=args.verbose_status)
    
    if meso_score is not None:
        print(f"MesoNet Score: {meso_score:.4f} (0=Real, 1=Fake)")
    else:
        print("MesoNet not used (No faces found or error)")
    
    # --- Run Temp-D3 ---
    # print("\nRunning Temp-D3 (XCLIP-16)...")
    d3_score = run_d3(video_path, verbose=args.verbose_status)
    if d3_score is not None:
        print(f"Temp-D3 Score: {d3_score:.4f} (Higher value ~ more likely Fake/Anomaly)")
    else:
        print("Temp-D3 failed to run.")

    # print("-" * 30)
    # print("Done.")

if __name__ == "__main__":
    main()
