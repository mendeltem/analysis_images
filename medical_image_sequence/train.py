import numpy as np
# from joblib import Parallel, delayed
import keras as ks
import numpy as np
#from bayes_opt import BayesianOptimization # conda install -c conda-forge bayesian-optimization

from pathlib import Path
from util import normalize_contrast, resize
import imageio

import math
import keras

from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization # conda install -c conda-forge bayesian-optimization
from keras.models import load_model

#from dataloader import DataLoader, EvalDataLoader
from dataloader import DataLoader
import random
from model import make_LSTM_model_singleInput,make_LSTM_model
from util import load_cases_labels
import json
#

import configparser
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
#

data = configparser.ConfigParser()
data.sections()
data.read('config.ini')

TRAIN_PATIENT_DIR       =  data["DATA"]["TRAIN_PATIENT_DIR"]
TRAIN_LABEL_PATH        =  data["DATA"]["TRAIN_LABEL_PATH"]


batch_size = 1 # https://github.com/keras-team/keras/issues/85


SEED  = 4
#sequence of picture would be provided
sequence_length = 20
#epochs
epochs = 2
#make images smaller
downsize = 8
#IMSIZE = tuple(int(ti/downsize) for ti in data["IMSIZE"])

IMSIZE  = (int(512/downsize),int(512/downsize))



def train_on_batch(model, generator, case, sequence_length, out = 1, pred = 1):
    """training on one case
    Arguments:
        model: cnn lstm model
        
        case:  (String) The name of the case
        
        sequence_length:  (int) The nummber of Images given each training step
                                One whole case consists of multiple sequences 

        out:  (bool)  print the output
        
        pred: (bool) predict the the images and print the output

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    mean_train_acc = []
    mean_train_loss = []

    sequence_index = 0
    #get the steping sice for one case
    sequence_steps = generator.__lenSequence__(case, sequence_length)
    
    
    print("Number of Images in this case : " + str(generator.get_n_images(case)))
    
    for sequence in range(sequence_steps):
        item, y = generator.__getSequence__(sequence_index, case, sequence_length)
        
        #sequence index goes up
        sequence_index += sequence_length
        #training only on sequences
        history = model.train_on_batch(item  , y) 

        history = ks.utils.generic_utils.to_list(history)
        
        mean_train_loss.append(history[0])        
        mean_train_acc.append(history[1])

  
    model.reset_states()      
    if out:    
        print("label:", y)
        print("train_loss: "+str(np.mean(mean_train_loss)))
        print("train_acc: "+ str(np.mean(mean_train_acc))) 
    
    return np.mean(mean_train_loss), np.mean(mean_train_acc)  
        
def testing_on_batch(model, generator, case, sequence_length, out = 1):
    """testing one case   
    """
    
    mean_test_acc = []
    mean_test_loss = []
    
    sequence_index = 0
    sequence_steps = generator.__lenSequence__(case, sequence_length)
    
    for sequence in range(sequence_steps):
        #get image sequenceses from this case with a length
        item, y = generator.__getSequence__(sequence_index, case, sequence_length)
        
        #sequence index goes up
        sequence_index += sequence_length
        #training only on sequences
              
        #testing 
        test_loss, test_acc = model.test_on_batch( item  , y)  
        
        mean_test_acc.append(test_acc)
        mean_test_loss.append(test_loss)
        
        
    model.reset_states()   
    if out:    
        print("label:", y)
        print("test_loss: "+str(mean_test_loss))
        print("test_acc: "+ str(np.mean(mean_test_acc)))
#        print("\n  prediction for this case:", (np.around(prediction[0][0],decimals = 4)))    
        
    return np.mean(mean_test_loss), np.mean(mean_test_acc)


def predicting_on_batch(model, generator, case, sequence_length, out = 1):
    """Predicting one case   
    """
    sequence_index = 0
    
    print("--------------------------------------------------------------------")
    print("Number of Images in this case : " + str(generator.get_n_images(case)))
    
    print("Case:" + case)
    
    sequence_steps = generator.__lenSequence__(case, sequence_length)
    
    for sequence in range(sequence_steps):
        #get image sequenceses from this case with a length
        item, y = generator.__getSequence__(sequence_index, case, sequence_length)
        
        #sequence index goes up
        sequence_index += sequence_length
        #training only on sequences
              
        #predict for debug
        prediction = model.predict_on_batch(item)
    model.reset_states()   
    if out:    
        print("prediction for this case:", (np.around(prediction[0][0],decimals = 4)))    
        
        print("Label for this case: " + str(y[0]))
        
    return prediction, y

#"""the Model"""
charite_model_gamma = make_LSTM_model(1,
                        sequence_length,
                        image_size=IMSIZE
                        )

#
##try dieffent models
#charite_model_gamma = load_model('charite_model_gamma.h5')
#

data_generator =  DataLoader(TRAIN_PATIENT_DIR,
                              TRAIN_LABEL_PATH,
                              sequence_length,
                              img_size=IMSIZE,
                              batch_size=int(batch_size),
                              n_splits = 10
                             )

input_shape = data_generator.get_inputshape()



charite_model = make_LSTM_model_singleInput(
                    batch_size=1,
                    sequence_length = sequence_length,
                    image_size=input_shape
                    )

#charite_model_gamma = make_LSTM_model(1,
#                        sequence_length,
#                        image_size=input_shape
#                        )

charite_model.summary()


train_cases = data_generator.get_train_cases()

test_cases = data_generator.get_test_cases()

#saving the history in log
log = open("train.log",'w')


#    
"""for debuging purpose we only use one loop from k fold split
it is now computationaly very hard to train on multiple cases at once

"""    

for epoch in range(epochs):
    print("Epoch:" + str(epoch))
    #wanna save the history
    train_loss_list  = []
    train_acc_list   = []
    
    test_loss_list  = []
    test_acc_list   = []    
    
    progbar = ks.utils.Progbar(len(train_cases))
    
    #goes through every sequence
    for case in train_cases:
        print("\n------------------------------------------------------------")
        print("case: ",case)
        
        mean_train_acc = []
        mean_train_loss = []
    
        sequence_index = 0
        #get the steping sice for one case
        sequence_steps = data_generator.__lenSequence__(case)
        
        print("Number of Images in this case : " + str(data_generator.get_n_images(case)))
        
        for sequence in range(sequence_steps):
            item, y = data_generator.__getSequence__(sequence_index, case)
            
            
            item.shape
            
            #sequence index goes up
            sequence_index += sequence_length
            #training only on sequences
            history = charite_model.train_on_batch(item  , y) 
    
            history = ks.utils.generic_utils.to_list(history)
            
            mean_train_loss.append(history[0])        
            mean_train_acc.append(history[1])
        
        charite_model.reset_states()      
    
        print("label:", y)
        print("train_loss: "+str(np.mean(mean_train_loss)))
        print("train_acc: "+ str(np.mean(mean_train_acc))) 
        log.write("\ntrain case: " 
          + str(case) 
          + "\n train_loss: " 
          +  str(np.mean(mean_train_loss)) 
          + "\n train_acc: " 
          + str(np.mean(mean_train_acc)) +
          "\n " +" \n")   
        
        train_loss_list.append(np.mean(mean_train_loss))
        train_acc_list.append(np.mean(mean_train_acc))
        progbar.add(1)
        
    progbar = ks.utils.Progbar(len(test_cases))   
    for case in test_cases:
        print("case: ",case)     
        test_loss, test_acc = testing_on_batch(charite_model,
                                               data_generator,
                                               case, 
                                               int(sequence_length/2),
                                               out = 1)
        log.write("\ntest case: " 
          + str(case) 
          + "\n train_loss: " 
          +  str(test_loss) 
          + "\n train_acc: " 
          + str(test_acc) +
          "\n " +" \n")           
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        progbar.add(1)
        
    print('______________________________________________')        
    print('accuracy training = {}'.format(np.mean(train_loss_list)))
    print('loss training = {}'.format(np.mean(train_loss_list))) 
    print('accuracy test = {}'.format(np.mean(test_acc_list)))
    print('loss test = {}'.format(np.mean(test_loss_list)))    
    
    
    
log.close()    
#   charite_model_gamma.save("charite_model_gamma.h5")
#    
    

#        
#item, y = generator.__getSequence__(sequence_index, train_cases[0], sequence_length)     
#
#prediction = model.predict_on_batch(item )
#print("prediction " + str(prediction))    
#print("y" +str(y))       
#        
#item, y = generator.__getSequence__(sequence_index, train_cases[1], sequence_length)     
#
#prediction = model.predict_on_batch(item )
#print("prediction " + str(prediction))    
#print("y" +str(y))    
#    train_loss_list.append(loss)
#
#
#print('______________________________________________')        
#print('accuracy training = {}'.format(np.mean(train_acc_list)))
#print('loss training = {}'.format(np.mean(train_loss_list))) 
#
#
#
#item, y = generator.__getitem__( case)
#
#print(len(item[0]))
#
#item, y = generator.__getitem_sequence__(16, case)
#
#print(len(item))
#
#
#
#
#model.predict(item)
#
#eval_network( lstm, 10 )
#
#def eval_network():
#    
#    batch_size = 10
#    
#    """the Model"""
#    model = make_LSTM_model(1,
#                            batch_size,
#                            image_size=IMSIZE
#                            )
#    
#    model.summary()
#    #loading all the labels
#    """loading all the cases and labels as string and int"""
#    cases, all_labels  = list(load_cases_labels(TRAIN_LABEL_PATH)) 
#    
#    generator =  DataLoader_beta(TRAIN_PATIENT_DIR,
#                                 cases,
#                                 all_labels, 
#                                 img_size=IMSIZE,
#                                 batch_size=int(batch_size))
#    
#    #k fold
#    k = 10
#    
#    boot = StratifiedKFold(n_splits=int(k))
#    
#    #if n_jobs > number of parallel units
#    rfc_score = np.array(Parallel(n_jobs=1)(delayed(processInput)(train,
#                                  test,
#                                  epochs,
#                                  cases,
#                                  generator,
#                                  model,
#                                  batch_size
#                                  ) for train,test in boot.split(cases, all_labels)))
#      
#    return rfc_score[-1]
#
#     
#def processInput(train, test,epochs, cases, generator,model, batch_size):
#    
#    train_cases  =  cases[train]
#    test_cases  =  cases[test]
#    
#    for epoch in range(epochs):
#        print("Epoche: ", epoch)
#        
#        mean_train_acc      = []
#        mean_train_loss     = []
#        mean_test_acc       = []
#        mean_test_loss      = []
#            
#    
#        for train_case in train_cases:
#            print("train_case", train_case)        
#            
#            stepsize = generator.__len__(train_case)
#            
#            batch_index = 0
#                 
#            for index in range(stepsize):
#        
#                item, y = generator.__getitem__(batch_index, train_case)
#                
#                batch_index += batch_size
#            
#                train_loss, train_acc = model.train_on_batch(   item  , y) 
#                print("label : ", y)
#                print("batch train_acc: ", train_acc)
#    
#            mean_train_acc.append(train_acc)
#            mean_train_loss.append(train_loss)
#            #print("train_loss for this case", np.mean(mean_train_loss))
#            print("train_acc for this case", train_acc)    
#                
#            model.reset_states()    
#            
#        print('______________________________________________')        
#        print('accuracy training = {}'.format(np.mean(mean_train_acc)))
#        print('loss training = {}'.format(np.mean(mean_train_loss))) 
#        
#        
#        for test_case in test_cases:
#            print("test case ", test_case)
#            stepsize = generator.__len__(test_case)
#            
#            batch_index = 0
#    
#            for index in range(stepsize):
#        
#                item, y = generator.__getitem__(batch_index, test_case)
#                
#                batch_index += batch_size
#            
#                test_loss, test_acc = model.test_on_batch(   item  ,y)  
#                
#                print("batch test accuracy: ", test_acc)
#                            
#            mean_test_acc.append(test_acc)
#            mean_test_loss.append(test_loss)
#            
#            print("test_acc for this case", test_acc) 
#            model.reset_states()    
#            
#        print('______________________________________________')        
#        print('accuracy test = {}'.format(np.mean(mean_test_acc)))
#        print('loss test = {}'.format(np.mean(mean_test_loss)))    
#        
#    return np.mean(mean_test_acc), np.mean(mean_test_loss)   
#         
#model.save("charite_model")

#generator =  DataLoader_beta(TRAIN_PATIENT_DIR,
#                             cases,
#                             all_labels, 
#                             img_size=IMSIZE,
#                             batch_size=int(10))

#
#if __name__=="__main__":
#    
#    with open("config.json") as data_file:
#        data = json.load(data_file)
#    
#    TRAIN_PATIENT_DIR       =  data["TRAIN_PATIENT_DIR"]
#    TRAIN_LABEL_PATH        = data["TRAIN_LABEL_PATH"]
#    IMSIZE                   = data["IMSIZE"]
#    
#    epochs = 2
#    
#    downsize = 4
#    IMSIZE = tuple(int(ti/downsize) for ti in IMSIZE)
#    
#    # example for parameter optimization, parameter raum mit Grenzen
#    pbounds = {'lstm': (1, 100)}
#    
#    optimizer = BayesianOptimization(
#        f=eval_network,
#        pbounds=pbounds,
#        random_state=1,
#    )
#    
#    optimizer.maximize(
#    init_points=4,
#    n_iter=5,
#    )
#    print("Optimal Values: ",optimizer.max)
#
#
#print("---------------------------------------------------------------------")




#case_to_predict = cases[0]
#
#item, y = generator.__getitem__(0, case_to_predict)
#
#train_loss, train_acc = model.train_on_batch(item  , [y[0]]) 
#
#
#import matplotlib.pyplot as plt
#plt.imshow(item[1][1])
#
#
#
#
#stepsize = generator.__len__(case_to_predict)
#
#item, y = generator.__getitem__(0, cases[0])
#
#item = np.array(item)
#

#case_to_predict = cases[0]
#
#stepsize = generator.__len__(case_to_predict)
#
#
#batch_index = 0
#
#for index in range(stepsize):
#
#    item, y = generator.__getitem__(batch_index, case_to_predict)
#
#    batch_index += batch_size
#    
##    dwi =  np.expand_dims(item[0], 0)   
##    flair = np.expand_dims(item[1], 0)   
#    
#    prediction = model.predict_on_batch(item)
#    
#label = y[0]
# 
#print("case: ", case_to_predict)    
#print("prediction: ", prediction)
##    
#batch_size = 10
#
#model = make_LSTM_model(1,
#                        batch_size,
#                        image_size=IMSIZE
#                        )
#    
##"""the Model"""
##model = make_model(60,IMSIZE, debug=True)
##loading all the labels
#"""loading all the cases and labels as string and int"""
#cases, all_labels  = list(load_cases_labels(TRAIN_LABEL_PATH)) 
#
#generator =  DataLoader_beta(TRAIN_PATIENT_DIR,
#                             cases,
#                             all_labels, 
#                             img_size=IMSIZE,batch_size=int(10))
#
#
#
#mean_train_acc      = []
#mean_train_loss     = []
#mean_test_acc       = []
#mean_test_loss      = []
#
#train_cases  =  cases[:20]
##
##test_cases  =  cases[test]
#
##
##for epoch in range(epochs):
##    print("Epoche: ", epoch)
#
#for train_case in train_cases:
#    print("train_case", train_case)        
#    
#    stepsize = generator.__len__(train_case)
#    
#    batch_index = 0
#         
#    for index in range(stepsize):
#
#        item, y = generator.__getitem__(batch_index, train_case)
#        batch_index += batch_size
#        #print(item[1].shape)
#        
#        
#    
#        train_loss, train_acc = model.train_on_batch(   item  , y) 
#        
#        
#        print("batch train_acc: ", train_acc)
#
#    mean_train_acc.append(train_acc)
#    mean_train_loss.append(train_loss)
#    #print("train_loss for this case", np.mean(mean_train_loss))
#    print("train_acc for this case", np.mean(mean_train_acc))    
#        
#    model.reset_states()    
#    
#print('______________________________________________')        
#print('accuracy training = {}'.format(np.mean(mean_train_acc)))
#print('loss training = {}'.format(np.mean(mean_train_loss))) 
    

#
#def get_layer_names(model):
#    layer_names = []
#    for layer in model.layers:
#        layer_names.append(layer.name)
#    return layer_names
#
#from keras import Model
#
#mixed_layer_names = get_layer_names(charite_model_gamma)
#
#main_mixed_layer_names = [ln for ln in mixed_layer_names]
#
#x = charite_model_gamma.input
#outs = []
#for ln in main_mixed_layer_names:
#    outs.append(charite_model_gamma.get_layer(ln).output)
#    
#layer_model = Model(inputs=x, outputs=outs)
#
#layer_model.summary()
#
#item, y = generator_gamma.__getSequence__(0, case, 20)
#
#layer_model.predict_on_batch(item)
#
#
#
#
#
#
#
    
#for case in cases[20:]:
#    predict = predicting_on_batch(charite_model_gamma, 
#                                  generator_gamma,
#                                  case, 
#                                  int(sequence_length/2), out = 1)

