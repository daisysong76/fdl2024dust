import pandas as pd
import joblib

# Load the original pickle files
video_test = pd.read_pickle('/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_image.pkl')
label_test = pd.read_pickle('/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_label.pkl')

# Save using joblib
joblib.dump(video_test, '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_image.joblib')
joblib.dump(label_test, '/home/daisysong/2024-HL-Virtual-Dosimeter/src/notebooks/test_label.joblib')
