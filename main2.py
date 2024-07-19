import numpy as np
import pandas as pd
from time import time
from joblib import dump
from sklearn import svm
from skimage.io import imread
from typing import List, Tuple
from datetime import timedelta
from math import pow, floor, log
from skimage.transform import resize
from os import listdir, path as path_
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class SVM:
    """
        Class for main
    """

    CATEGORIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    TRAIN_DIR = './USPS_images/images/train'
    TEST_DIR = './USPS_images/images/test'

    def __init__(self) -> None:
        """
            Constructor for Main class,
        """

        self.x_test, self.y_test = self.read_test_files()
        self.x_train, self.y_train = self.read_train_files()

    @staticmethod
    def convert_size(size_bytes: int) -> str:
        """
            Method to convert byte size to KB - MB - GB,

            Parameters:
                size_bytes (int): The size in bytes

            Returns:
                The result of the conversion
        """

        if not size_bytes:
            return "0B"

        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")

        i = int(floor(log(size_bytes, 1024)))
        s = round(size_bytes / pow(1024, i), 2)

        return "%s %s" % (s, size_name[i])

    def read_test_files(self) -> Tuple[List, List]:
        """
            Method to read the test image files,

            Returns:
                The x_test and y_test lists
        """

        flat_data_arr = []  # input array
        target_arr = []  # output array

        print("-" * 10 + " Test Reading " + "-" * 10 + "\n")

        # path which contains all the categories of images
        for i in self.CATEGORIES:
            print(f'+ loading... category : {i}')

            for img in [j for j in listdir(self.TEST_DIR) if j.startswith(str(i))]:
                img_array = imread(path_.join(self.TEST_DIR, img))

                # resize images to make them all the same size
                img_resized = resize(img_array, (150, 150, 3))

                flat_data_arr.append(img_resized.flatten())

                target_arr.append(i)

        # create aray of flat data
        flat_data = np.array(flat_data_arr)
        print(f"\n ++ All categories loaded: {len(flat_data)} files !\n")

        # create the dataframe
        target = np.array(target_arr)
        df = pd.DataFrame(flat_data)

        df['Target'] = target

        x = df.iloc[:, :-1]  # input data
        y = df.iloc[:, -1]  # output data

        return x, y

    def read_train_files(self) -> Tuple[List, List]:
        """
            Method to read the train image files,

            Returns:
                The x_train and y_train lists
        """

        # to reduce time of training, we use a variable to find the x entries of any numbers,
        # EX: if TRAIN_NUMBER = 10 >>> we have 10 images for number "1", 10 images for number "2" and ...
        train_number = 10

        flat_data_arr = []  # input array
        target_arr = []  # output array

        print("-" * 10 + " Train Reading " + "-" * 10 + "\n")

        # path which contains all the categories of images
        for i in self.CATEGORIES:
            print(f'+ loading... category : {i}')

            for img in [j for j in listdir(self.TRAIN_DIR) if j.startswith(str(i))][:train_number]:
                img_array = imread(path_.join(self.TRAIN_DIR, img))  # , as_gray=True)

                # resize images to make them all the same size
                img_resized = resize(img_array, (150, 150, 3))

                flat_data_arr.append(img_resized.flatten())

                target_arr.append(i)

        # create aray of flat data
        flat_data = np.array(flat_data_arr)
        print(f"\n ++ All categories loaded: {len(flat_data)} files !\n")

        # create the dataframe
        target = np.array(target_arr)
        df = pd.DataFrame(flat_data)

        df['Target'] = target

        x = df.iloc[:, :-1]  # input data
        y = df.iloc[:, -1]  # output data

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.20, random_state=30, stratify=y)

        return x_train, y_train

    def create_model(self, param_grid: dict) -> None:
        """
            Method to create model from parameters,

            Parameters:
                param_grid (dict): The parameters for GridSearchCV
        """

        print(f"Kernel: {param_grid['kernel']} - C: {param_grid['C']} - Gamma: {param_grid['gamma']}")

        # make the model
        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, param_grid)

        # train model
        start_time = time()
        model.fit(self.x_train, self.y_train)
        end_time = time()

        # save the model
        dump(model, 'model.pkl')

        b = path_.getsize("model.pkl")

        print(f"Time: {str(timedelta(seconds=end_time - start_time))} - Size: {self.convert_size(b)}")

        # predict the tests
        y_pred = model.predict(self.x_test)

        print(f"The model is {accuracy_score(y_pred, self.y_test) * 100}% accurate !\n")

    def run_tests(self) -> None:
        """
            Method to run tests,
        """

        keeper = [
            {'C': [1, 100], 'gamma': [0.0001, 1], 'kernel': ['sigmoid']},
            {'C': [1, 100], 'gamma': [0.0001, 1], 'kernel': ['linear']},
            {'C': [1, 100], 'gamma': [0.0001, 1], 'kernel': ['poly']},
            {'C': [1, 100], 'gamma': [0.0001, 1], 'kernel': ['rbf']},

            {'C': [1, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['sigmoid']},
            {'C': [1, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['linear']},
            {'C': [1, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['poly']},
            {'C': [1, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['rbf']},

            {'C': [1, 10, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['sigmoid']},
            {'C': [1, 10, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['linear']},
            {'C': [1, 10, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['poly']},
            {'C': [1, 10, 100], 'gamma': [0.0001, 0.1, 1], 'kernel': ['rbf']},
        ]

        print("-" * 10 + " Create Models " + "-" * 10 + "\n")

        for i in keeper:
            print(f'{keeper.index(i)})', end=" ")
            self.create_model(i)


if __name__ == "__main__":
    svm_ = SVM()
    svm_.run_tests()
