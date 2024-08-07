import pickle

def save_to_pkl(data, labels, video_output_path, label_output_path):
    with open(video_output_path, 'wb') as f:
        pickle.dump(data, f)
    with open(label_output_path, 'wb') as f:
        pickle.dump(labels, f)

def load_data(video_path, label_path):
    with open(video_path, 'rb') as f:
        video_data = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return video_data, labels
