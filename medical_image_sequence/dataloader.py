import tensorflow as tf


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
#from bayes_opt import BayesianOptimization # conda install -c conda-forge bayesian-optimization
import imageio
from pathlib import Path
from util import normalize_contrast, resize

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import math
import keras


class DataLoader(keras.utils.Sequence):
    """
    
    """

    def __init__(self, 
                 data_path:Path, 
                 label_path:Path,
                 sequence_length,
                 img_size=(512,512),
                 batch_size=1,
                 n_splits=10
                 ):
        
        self.batch_size             = batch_size
        self.img_size               = img_size
        self.input_shape            = (img_size[0] , img_size[1] * 2)
        self.label_path             = label_path
        self.data_path              = data_path
        self.sequence_lenght        = sequence_length
        
        self.cases,  self.labels   =  list(self.load_cases_labels(label_path)) 
        
        
        self.cases,  self.labels = self.shuffle()
        self.zipped_list = []
        self.data = {}

        for case, label in zip(self.cases, self.labels):
            case_paths = self.load_image_paths(case)
            self.data[case] = []
            self.data[case].extend([case_paths, label]) 
            
            
        boot = StratifiedKFold(n_splits)
        
        
        for train,test in boot.split(self.cases, self.labels):
            self.train_cases  =  self.cases[train]
            self.test_cases  =  self.cases[test]
    
    
    def get_train_cases(self):
        return self.train_cases 
    
    
    def get_test_cases(self):
        return self.test_cases
    
    
    def __getSequence__(self, idx, case):
        concateinated_images   = []   
        y       = []
        
        #Image path from given case  
        image_paths_from_case = self.data[case][0]
        #image label given case
        label = self.data[case][1]
        
        #image paths given batch size
        image_paths_as_sequence = image_paths_from_case[idx: idx + self.sequence_lenght]
             
        for image in image_paths_as_sequence:
            dwi_path   = image[0]
            flair_path = image[1] 
            #load images and resize and normalize
            dwi_image = normalize_contrast(
                    resize(
                    imageio.imread(dwi_path),
                    self.img_size )
                    )
            flair_image = normalize_contrast(
                    resize(
                    imageio.imread(flair_path),
                    self.img_size )
                    )
            dwi_image_with_3_channel = np.zeros(  dwi_image.shape + (3,))              
            dwi_image_with_3_channel[:,:,0] =   dwi_image  
            dwi_image_with_3_channel[:,:,1] =   dwi_image  
            dwi_image_with_3_channel[:,:,2] =   dwi_image  
            
            flair_image_with_3_channel = np.zeros(  flair_image.shape + (3,))        
            flair_image_with_3_channel[:,:,0] =   flair_image  
            flair_image_with_3_channel[:,:,1] =   flair_image  
            flair_image_with_3_channel[:,:,2] =   flair_image    
                         
       
            concat_image = np.concatenate( [dwi_image_with_3_channel,
                                            flair_image_with_3_channel],
                                             axis = 1)
            concateinated_images.append(concat_image)
            
            y.append(int(label[0]))
            

        concateinated_images_exp = np.expand_dims(np.array(concateinated_images),
                                                  0) 

        return concateinated_images_exp, [y[0]] 
    
        
            
    def get_inputshape(self):
        return self.input_shape        
            
    def get_n_images(self, case):
        return len(self.data[case][0])        
  
    def get_list(self, case):
        return self.data
            
    def load_image_paths(self, case_name):
        
        zipped_list = []
        """rename caes like the file directory"""
        case_preprocessed = case_name.replace("Set","Set_")
        case_preprocessed = case_preprocessed.replace("Case", "/Case_")

        dwi_ip = list(Path(str(self.data_path)+ str(case_preprocessed)+"/"+ "DWI").glob("**/*.dcm"))
        flair_ip = list(Path(str(self.data_path)+ str(case_preprocessed)+"/"+ "FLAIR").glob("**/*.dcm"))
        
        """there are different count of pictures, we change it so that has the same
        count. (We duplicate the missing pictures from the most pictures )
        """
        if len(dwi_ip) > len(flair_ip):
            jump = int(np.round(len(dwi_ip) / len(flair_ip)))
            #print("Jump left",jump )
    
            """check if its around"""
            mod = len(dwi_ip) % len(flair_ip)
            for i in range(len(dwi_ip)-mod):
                zipped_list.append((dwi_ip[i], flair_ip[int(i/ jump)]))
                     
        elif len(dwi_ip) < len(flair_ip):
            #print("Jump right",jump )
            jump = int(np.round(len(flair_ip) / len(dwi_ip)))
            mod = len(flair_ip) % len(dwi_ip)
            for i in range(len(flair_ip) - mod):
                zipped_list.append((dwi_ip[int(i/ jump)], flair_ip[i]))
        else:
            #print("No juump")
            zipped_list = list(zip(dwi_ip,flair_ip))
                         
        return zipped_list
  
            
    def __len__(self, case):
        return math.ceil(len(self.data[case][0]) / self.batch_size ) 
    
    def __lenSequence__(self, case):
        return math.ceil(len(self.data[case][0]) / self.sequence_lenght ) 
    
    

    
    def __getitem__(self, case):
        x_dwi   = []   
        x_flair = []
        y       = []
        
        #Image path from given case  
        image_paths_from_case = self.data[case][0]
        
        #image label given case
        label = self.data[case][1]
        
        
        for image in image_paths_from_case:
            dwi_path   = image[0]
            flair_path = image[1] 
            #load images and resize and normalize
            dwi_image = normalize_contrast(
                    resize(
                    imageio.imread(dwi_path),
                    self.img_size )
                    )
            flair_image = normalize_contrast(
                    resize(
                    imageio.imread(flair_path),
                    self.img_size )
                    )
            dwi_image_with_3_channel = np.zeros(  dwi_image.shape + (3,))              
            dwi_image_with_3_channel[:,:,0] =   dwi_image  
            dwi_image_with_3_channel[:,:,1] =   dwi_image  
            dwi_image_with_3_channel[:,:,2] =   dwi_image  
            
            flair_image_with_3_channel = np.zeros(  flair_image.shape + (3,))        
            flair_image_with_3_channel[:,:,0] =   flair_image  
            flair_image_with_3_channel[:,:,1] =   flair_image  
            flair_image_with_3_channel[:,:,2] =   flair_image    
                         
       
            x_dwi.append(dwi_image_with_3_channel)
            x_flair.append(flair_image_with_3_channel)
            
            y.append(int(label[0]))
            
        while len(x_dwi) % self.batch_size != 0:
            x_dwi.append(np.zeros(  dwi_image.shape + (3,))  )    
            x_flair.append(np.zeros(  dwi_image.shape + (3,))  )
            
            
        x_dwi_images   = np.expand_dims(np.array(x_dwi), 0) 
        x_flair_images = np.expand_dims(np.array(x_flair), 0) 
        
        return [x_dwi_images, x_flair_images], [y[0]]
    
    
    
    def load_cases_labels(self, path:Path):
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

        
    def shuffle(self):
        
        
        cases = self.cases.reshape(self.cases.shape[0],1)
        cases_labels = np.concatenate([cases, self.labels], axis = 1)
        
        np.random.shuffle(cases_labels)
    
        return cases_labels[:,0], cases_labels[:,1]
        