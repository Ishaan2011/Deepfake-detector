import numpy as np
import sys
from classifiers import *
from pipeline import FaceFinder, FaceBatchGenerator, predict_faces
from math import floor

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_F2F.h5')

# 2 - Specify your video path here OR pass it as a command line argument
#     Usage: python example_video.py path/to/your/video.mp4

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    # Default video path - change this to your video
    video_path = '../Deepfake_testing/sa1-video-mccs0.mp4'

print(f'Analyzing video: {video_path}')

# 3 - Extract faces from video frames
# frame_subsample_count controls how many frames to analyze (higher = more accurate but slower)
frame_subsample_count = 30

face_finder = FaceFinder(video_path, load_first_face=True)
skipstep = max(floor(face_finder.length / frame_subsample_count), 0)

print(f'Video has {face_finder.length} frames at {face_finder.fps} FPS')
print(f'Extracting faces (analyzing every {skipstep + 1} frames)...')

face_finder.find_faces(resize=0.5, skipstep=skipstep)

print(f'Found faces in {len(face_finder.coordinates)} frames')

# 4 - Predict deepfake on extracted faces
print('Running deepfake detection...')
generator = FaceBatchGenerator(face_finder)
predictions = predict_faces(generator, classifier)

# 5 - Aggregate results
if len(predictions) > 0:
    mean_prediction = np.mean(predictions)
    fake_percentage = np.mean(predictions > 0.5) * 100
    
    print(f'\n=== Results ===')
    print(f'Frames analyzed: {len(predictions)}')
    print(f'Average prediction score: {mean_prediction:.4f}')
    print(f'Frames detected as fake: {fake_percentage:.1f}%')
    print(f'\nFinal verdict: {"FAKE (Deepfake detected)" if mean_prediction > 0.5 else "REAL (No deepfake detected)"}')
else:
    print('No faces found in the video!')