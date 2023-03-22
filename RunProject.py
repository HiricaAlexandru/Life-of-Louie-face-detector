from Parameters import *
from FacialDetector import *
import extragere_fete as ef
import os


params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.overlap = 0.3

facial_detector: FacialDetector = FacialDetector(params)


if not(os.path.exists("rezultate")):
    os.mkdir("rezultate")
    os.mkdir("rezultate\\task1")
    os.mkdir("rezultate\\task2")

positive_features, positive_features_labels, negative_features = ef.load_features()

training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(positive_features_labels.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifierMLP(training_examples, train_labels)

detections, scores, file_names = facial_detector.run()

np.save("rezultate\\task1\\detections_all_faces.npy", detections)
np.save("rezultate\\task1\\file_names_all_faces.npy", file_names)
np.save("rezultate\\task1\\scores_all_faces.npy", scores)

##task2

all_training_characters = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
all_training_characters_labels = np.concatenate((np.squeeze(positive_features_labels), np.zeros(negative_features.shape[0])), axis = 0)


facial_detector.train_classifierMLPCharacterRecognition(all_training_characters, all_training_characters_labels)

all_detections_packed, all_scores_packed, all_files_packed = facial_detector.run_task2()

detections_andy, detections_louie, detections_ora, detections_tommy = all_detections_packed
scores_andy, scores_louie, scores_ora, scores_tommy = all_scores_packed
file_names_andy, file_names_louie, file_names_ora, file_names_tommy = all_files_packed

np.save("rezultate\\task2\\detections_andy.npy", detections_andy)
np.save("rezultate\\task2\\file_names_andy.npy", file_names_andy)
np.save("rezultate\\task2\\scores_andy.npy", scores_andy)

np.save("rezultate\\task2\\detections_louie.npy", detections_louie)
np.save("rezultate\\task2\\file_names_louie.npy", file_names_louie)
np.save("rezultate\\task2\\scores_louie.npy", scores_louie)

np.save("rezultate\\task2\\detections_ora.npy", detections_ora)
np.save("rezultate\\task2\\file_names_ora.npy", file_names_ora)
np.save("rezultate\\task2\\scores_ora.npy", scores_ora)

np.save("rezultate\\task2\\detections_tommy.npy", detections_tommy)
np.save("rezultate\\task2\\file_names_tommy.npy", file_names_tommy)
np.save("rezultate\\task2\\scores_tommy.npy", scores_tommy)
