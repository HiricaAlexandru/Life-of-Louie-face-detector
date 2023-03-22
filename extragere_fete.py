import cv2 as cv
from skimage.feature import hog
import os
import Parameters as params
import numpy as np

path_date_antrenare_relativ = "antrenare"

def return_character_embeding(name):
    if name == 'unknown':
        return 0
    elif name == 'andy':
        return 1
    elif name == 'ora':
        return 2
    elif name == 'louie':
        return 3
    elif name == 'tommy':
        return 4

def return_character_from_embeding(number):
    if number == 0:
        return 'unknown'
    elif number == 1:
        return 'andy'
    elif number == 2:
        return 'ora'
    elif number == 3:
        return 'louie'
    elif number == 4:
        return 'tommy'

def extract_window_of_character(character_name):
    print("Extrag fereastra pentru: ", character_name)
    path_character_images = f"{path_date_antrenare_relativ}\\{character_name}"
    path_annotations = f"{path_date_antrenare_relativ}\\{character_name}_annotations.txt"

    x = []
    y = []

    f = open(path_annotations,'r')
    current_image = None
    line = f.readline()
    while line:
        line = line.rstrip().split(" ")

        if current_image != line[0]:
            img = cv.imread(f"{path_character_images}\\{line[0]}", cv.IMREAD_GRAYSCALE)

        current_image = line[0]
        x_min, y_min, x_max, y_max = int(line[1]), int(line[2]), int(line[3]), int(line[4]) 
        window = img[y_min:y_max, x_min:x_max].copy()
        window = cv.resize(window, (params.dim_window, params.dim_window))
        features = hog(window, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
        x.append(features)
        number_embeding = int(return_character_embeding(line[5]))
        y.append(number_embeding)

        (h, w) = window.shape[:2]
        (cx, cy) = (w//2, h//2)

        for angle in np.arange(-5,5.5,0.5):
            rotation_matrix = cv.getRotationMatrix2D((cx,cy), angle, 1.0)
            rotated_image = cv.warpAffine(window, rotation_matrix, (w,h))
            features = hog(rotated_image, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=True)

            #cv.imshow("sa",rotated_image)
            #cv.waitKey(0)
            x.append(features)
            y.append(number_embeding)

        features = hog(np.fliplr(window), pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                        cells_per_block=(2, 2), feature_vector=True)

        x.append(features)
        y.append(number_embeding)

        line = f.readline() 
    
    x = np.array(x)
    y = np.array(y)

    return x,y

def get_intersection_of_boxes(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    return inter_area

def get_box_coordinates(name):
    path_annotations = f"{path_date_antrenare_relativ}\\{name}_annotations.txt"

    dictionar_coord = dict()

    f = open(path_annotations,'r')
    current_image = None
    line = f.readline()
    while line:
        line = line.rstrip().split(" ")

        if line[0] in dictionar_coord.keys():
            dictionar_coord[line[0]].append((int(line[1]), int(line[2]), int(line[3]), int(line[4])))
        else:
            dictionar_coord[line[0]] = [(int(line[1]), int(line[2]), int(line[3]), int(line[4]))]

        line = f.readline()

    return dictionar_coord

def create_negative_features():

    print("Generez exemple negative")
    negative_features = []
    resize = [1, 0.9, 0.75, 0.5]

    for name in ['andy', 'louie', 'ora', 'tommy']:
        path_character_images = f"{path_date_antrenare_relativ}\\{name}"
        
        box_coordinates = get_box_coordinates(name)

        for image in box_coordinates.keys():
            img = cv.imread(f"{path_character_images}\\{image}", cv.IMREAD_GRAYSCALE)

            for resize_factor in resize:
                img_resized = cv.resize(img, None, fx = resize_factor, fy = resize_factor, interpolation = cv.INTER_CUBIC)

                num_rows = img_resized.shape[0]
                num_cols = img_resized.shape[1]
                
                if num_cols >= params.dim_window and num_rows >= params.dim_window:

                    for i in range(13):
                        #select 13 random windows
                        
                        x = np.random.randint(low=0, high=num_cols - params.dim_window)
                        y = np.random.randint(low=0, high=num_rows - params.dim_window)

                        x_min = x
                        y_min = y
                        x_max = x + params.dim_window
                        y_max = y + params.dim_window

                        ok = True

                        for bbox in box_coordinates[image]:
                            if get_intersection_of_boxes(bbox, (int(x_min / resize_factor), int(y_min/ resize_factor), int(x_max/ resize_factor), int(y_max / resize_factor))) > 0:
                                ok = False
                        img_copy = img.copy()
                        if ok == True:
                            window = img_resized[y_min:y_max, x_min:x_max].copy()
                            #cv.rectangle(img_copy, (int(x_min / resize_factor), int(y_min/ resize_factor)), (int(x_max/ resize_factor), int(y_max / resize_factor)), (0, 0, 255), thickness=1)
                            #cv.imshow("as", img_copy)
                            #cv.waitKey(0)
                            features = hog(window, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=True)
                            negative_features.append(features)

    negative_features = np.array(negative_features)
    return negative_features

        
def load_features():
    positive_features_path = "features\\positive_features.npy"
    negative_features_path = "features\\negative_features.npy"
    positive_labels_path = "features\\positive_labels.npy"

    positive_features = None
    positive_features_labels = None
    negative_features = None

    if os.path.exists("features"):
        positive_features = np.load(positive_features_path, allow_pickle=True)
        positive_features_labels = np.load(positive_labels_path, allow_pickle=True)
        negative_features = np.load(negative_features_path, allow_pickle=True)
        print(positive_features.shape, positive_features_labels.shape, negative_features.shape)
    else:
        os.mkdir("features")
        negative_features = create_negative_features()
        x_andy, y_andy = extract_window_of_character("andy")
        x_louie, y_louie = extract_window_of_character("louie")
        x_ora, y_ora = extract_window_of_character("ora")
        x_tommy, y_tommy = extract_window_of_character("tommy")

        positive_features = np.concatenate((x_andy, x_louie, x_ora, x_tommy),axis = 0)
        positive_features_labels = np.concatenate((y_andy, y_louie, y_ora, y_tommy),axis = 0)
        print(positive_features.shape, positive_features_labels.shape, negative_features.shape)
        
        np.save(negative_features_path, negative_features)
        np.save(positive_features_path, positive_features)
        np.save(positive_labels_path, positive_features_labels)

    return positive_features, positive_features_labels, negative_features
