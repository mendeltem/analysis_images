import imageio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

def load_next_image(image_path:Path):
    image_path = Path(image_path)
    images = list(image_path.glob("**/*.dcm"))
    while images:
        yield imageio.imread(str(images[0]))
        del images[0]

def load_labels(path:Path):
    path = Path(path)
    dfs = pd.read_excel(path)
    dfs = dfs.iloc[:,[0,-1]]
    labels = list(dfs.values)
    while labels:
        yield labels[0]
        del labels[0]

def load_labels_only(path:Path):
    path = Path(path)
    dfs = pd.read_excel(path)
    dfs = dfs.iloc[:,[-1]]
    return list(dfs.values)


def load_cases_labels(path:Path):
    """load case names and labels from xml
    rename the case name according to dir name
    """
    
    renamed_cases = []
    path = Path(path)
    dfs = pd.read_excel(path)
    dfs = dfs.iloc[:,[0,-1]]

    labels = dfs.iloc[:,[-1]].values
    cases = dfs.iloc[:,[0]].values

    """rename the single cases like case1 to case01"""
    
    for case in cases:
        if len(re.search("(\d+)(?!.*\d)", case[0])[0]) < 2:
            new_tail = "0" + re.search("(\d+)(?!.*\d)", case[0])[0]
            renamed_cases.append(re.sub(r"(\d+)(?!.*\d)", new_tail, case[0])) 
        else:
            renamed_cases.append(case[0])
            
    return np.array(renamed_cases), labels


def load_data(data_path:Path, label_path:Path):
    data_path = Path(data_path)
    label_path = Path(label_path)
    for case_name, label in load_labels(label_path):
        for dwi_image, flair_image in zip(
                load_next_image(data_path / case_name / "DWI"),
                load_next_image(data_path / case_name / "FLAIR")):
                yield (dwi_image, flair_image), label


def normalize_contrast(image:np.array):
    return (image - np.min(image)) / (np.max(image)-np.min(image))
    # return cv2.normalize(image, None)

def resize(image, size:tuple):
    return cv2.resize(image, size)

#if __name__=="__main__":
#    print(load_labels("./Answers_for_the_training_tool.xlsx"))
#    image_tup = []
#    for tup in load_data("./","./Answers_for_the_training_tool.xlsx"):
#        image_tup.append(tup)
#    dwi_image = normalize_contrast(resize(image_tup[len(image_tup)//2][0][0],(512,512)))
#    plt.imshow(dwi_image)
#    plt.show()


def get_dcm_paths(start_dir, extensions = ['dcm']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths


def rounding(digits,threshold = 0.50):
    """rounds a list of float digits with ad threshhold"""
    if type(digits) == list  or  type(digits) == np.ndarray:
        return np.array(list(map(rounding, digits)))
    if type(digits)== np.float64 or type(digits)== np.float32 or type(digits)== np.float:
        k = digits % 1
        f = digits - k

        if k >= threshold:
            return int(f + 1)
        else:
            return int(f)
    else:
        print(type(digits))
        raise ValueError("Wrong Type")