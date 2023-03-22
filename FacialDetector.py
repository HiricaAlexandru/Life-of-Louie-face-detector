from Parameters import *
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
import extragere_fete as ef
import os

class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        self.best_model_characters = None

    def train_classifierMLP(self, training_examples, train_labels):
        mlp_save_folder = "best_model"

        if not(os.path.exists(mlp_save_folder)):
            os.mkdir(mlp_save_folder)
            
        if os.path.exists(f"{mlp_save_folder}\\mlp{self.params.dim_hog_cell}.txt"): 
            self.best_model = pickle.load(open(f"{mlp_save_folder}\\mlp{self.params.dim_hog_cell}.txt", 'rb'))
            return

        best_accuracy = 0

        best_model = None
        print("Antrenez model pentru detectare faciala")
        model = MLPClassifier(hidden_layer_sizes=(100), verbose=True)
        model.fit(training_examples, train_labels)
        acc = model.score(training_examples, train_labels)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = deepcopy(model)
        
        # salveaza clasificatorul
        pickle.dump(best_model, open(f"{mlp_save_folder}\\mlp{self.params.dim_hog_cell}.txt", 'wb'))
        self.best_model = best_model

    def train_classifierMLPCharacterRecognition(self, training_examples, train_labels):
        mlp_save_folder = "best_model"

        if not(os.path.exists(mlp_save_folder)):
            os.mkdir(mlp_save_folder)
            
        if os.path.exists(f"{mlp_save_folder}\\mlpCharacters{self.params.dim_hog_cell}.txt"): 
            self.best_model_characters = pickle.load(open(f"{mlp_save_folder}\\mlpCharacters{self.params.dim_hog_cell}.txt", 'rb'))
            return

        best_accuracy = 0

        best_model = None
        print("Antrenez pentru clasificarea persoanelor")
        model = MLPClassifier(hidden_layer_sizes=(500,250,100,50), max_iter=200,verbose=True)
        model.fit(training_examples, train_labels)
        acc = model.score(training_examples, train_labels)
        print(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = deepcopy(model)
        
        # salveaza clasificatorul
        pickle.dump(best_model, open(f"{mlp_save_folder}\\mlpCharacters{self.params.dim_hog_cell}.txt", 'wb'))
        self.best_model_characters = best_model

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


    def run(self, negative_mining = False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')

        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista

        num_test_images = len(test_files)
        descriptors_to_return = []
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            resized = [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2]
            image_scores = []
            image_detections = []

            for resize in resized:
                img_resized = cv.resize(img, None, fx = resize, fy = resize, interpolation = cv.INTER_CUBIC)

                hog_descriptors = hog(img_resized, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                    cells_per_block=(2, 2), feature_vector=False)
                num_cols = img_resized.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img_resized.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template):
                    for x in range(0, num_cols - num_cell_in_template):
                        descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                        prediction = self.best_model.predict_log_proba([descr])[0]
                        score = np.max(prediction)
                        prediction_label = np.argmax(prediction)

                        if prediction_label == 1 and score > -4:
                            x_min = int((x * self.params.dim_hog_cell) / resize)
                            y_min = int((y * self.params.dim_hog_cell) / resize)
                            x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) /resize)
                            y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) / resize)
                            #img_crop = img[y_min:y_max, x_min: x_max]
                            #img_crop = cv.resize(img_crop, (self.params.dim_window, self.params.dim_window), interpolation = cv.INTER_CUBIC)


                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores), img.shape)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)

                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def run_task2(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        print("Run task 2")
        
        detections_andy = None  
        detections_ora = None  
        detections_louie = None  
        detections_tommy = None  

        scores_louie = np.array([])
        scores_andy = np.array([])
        scores_tommy = np.array([])
        scores_ora = np.array([])

        file_names_andy = np.array([])
        file_names_louie = np.array([])
        file_names_ora = np.array([])
        file_names_tommy = np.array([])

        # detectie din imagine, numele imaginii va aparea in aceasta lista

        num_test_images = len(test_files)
        descriptors_to_return = []
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            resized = [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2]
            image_scores_ora = []
            image_detections_ora = []

            image_scores_tommy = []
            image_detections_tommy = []

            image_scores_louie = []
            image_detections_louie = []

            image_scores_andy = []
            image_detections_andy = []

            for resize in resized:
                img_resized = cv.resize(img, None, fx = resize, fy = resize, interpolation = cv.INTER_CUBIC)

                hog_descriptors = hog(img_resized, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                    cells_per_block=(2, 2), feature_vector=False)
                num_cols = img_resized.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img_resized.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template):
                    for x in range(0, num_cols - num_cell_in_template):
                        descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                        prediction = self.best_model_characters.predict_log_proba([descr])[0]
                        score = np.max(prediction)
                        prediction_label = np.argmax(prediction)

                        x_min = int((x * self.params.dim_hog_cell) / resize)
                        y_min = int((y * self.params.dim_hog_cell) / resize)
                        x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) /resize)
                        y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) / resize)

                        if score > -4 and np.mean(img_resized.flatten()) > 80:

                            if prediction_label == 1:
                                image_detections_andy.append([x_min, y_min, x_max, y_max])
                                image_scores_andy.append(score)
                                #print("ANDY")

                            if prediction_label == 2:
                                image_detections_ora.append([x_min, y_min, x_max, y_max])
                                image_scores_ora.append(score)
                                #print("ORA")

                            if prediction_label == 3:
                                image_detections_louie.append([x_min, y_min, x_max, y_max])
                                image_scores_louie.append(score)
                                #print("LOUIE")

                            if prediction_label == 4:
                                image_detections_tommy.append([x_min, y_min, x_max, y_max])
                                image_scores_tommy.append(score)
                

            if len(image_scores_andy) > 0:
                image_detections_andy, image_scores_andy = self.non_maximal_suppression(np.array(image_detections_andy),
                                                                           np.array(image_scores_andy), img.shape)
            if len(image_scores_tommy) > 0:
                image_detections_tommy, image_scores_tommy = self.non_maximal_suppression(np.array(image_detections_tommy),
                                                                            np.array(image_scores_tommy), img.shape)

            if len(image_scores_louie) > 0:
                image_detections_louie, image_scores_louie = self.non_maximal_suppression(np.array(image_detections_louie),
                                                                            np.array(image_scores_louie), img.shape)

            if len(image_scores_ora) > 0:
                image_detections_ora, image_scores_ora = self.non_maximal_suppression(np.array(image_detections_ora),
                                                                            np.array(image_scores_ora), img.shape)
            #andy
            if len(image_scores_andy) > 0:
                if detections_andy is None:
                    detections_andy = image_detections_andy
                else:
                    detections_andy = np.concatenate((detections_andy, image_detections_andy))
                scores_andy = np.append(scores_andy, image_scores_andy)

                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores_andy))]
                file_names_andy = np.append(file_names_andy, image_names)

            #ora
            if len(image_scores_ora) > 0:
                if detections_ora is None:
                    detections_ora = image_detections_ora
                else:
                    detections_ora = np.concatenate((detections_ora, image_detections_ora))
                scores_ora = np.append(scores_ora, image_scores_ora)

                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores_ora))]
                file_names_ora = np.append(file_names_ora, image_names)

            #tommy
            if len(image_scores_tommy) > 0:
                if detections_tommy is None:
                    detections_tommy = image_detections_tommy
                else:
                    detections_tommy = np.concatenate((detections_tommy, image_detections_tommy))
                scores_tommy = np.append(scores_tommy, image_scores_tommy)

                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores_tommy))]
                file_names_tommy = np.append(file_names_tommy, image_names)

            #louie     
            if len(image_scores_louie) > 0:
                if detections_louie is None:
                    detections_louie = image_detections_louie
                else:
                    detections_louie = np.concatenate((detections_louie, image_detections_louie))
                scores_louie = np.append(scores_louie, image_scores_louie)

                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores_louie))]
                file_names_louie = np.append(file_names_louie, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                % (i, num_test_images, end_time - start_time))
            end_time = timeit.default_timer()

        
        detections_andy = np.array(detections_andy)
        detections_louie = np.array(detections_louie)
        detections_ora = np.array(detections_ora)
        detections_tommy = np.array(detections_tommy)
        
        all_detections_packed = (detections_andy, detections_louie, detections_ora, detections_tommy)
        all_scores_packed = (scores_andy, scores_louie, scores_ora, scores_tommy)
        all_files_packed = (file_names_andy, file_names_louie, file_names_ora, file_names_tommy)

        return all_detections_packed, all_scores_packed, all_files_packed

