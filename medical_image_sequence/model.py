import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Flatten, RepeatVector, Reshape
from keras.applications import VGG19
from keras.layers.merge import Concatenate
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam


"""we modified the video classifer 
model
https://riptutorial.com/keras/example/29812/vgg-16-cnn-and-lstm-for-video-classification

train on batch statefull
http://philipperemy.github.io/keras-stateful-lstm/
"""

def make_vgg(in_shape,name,output_layer=-1):
    vgg = VGG19(
            include_top=False, 
            input_shape=in_shape, 
            weights="imagenet")  
    cnn_out = GlobalAveragePooling2D()(vgg.layers[16].output)
    model = Model(vgg.input, cnn_out, name=f"VGG19_{name}")
    #model.trainable = False
    #model.summary()
    return model
 
def make_LSTM_model(batch_size =1,look_back=10,image_size=(512, 512), debug=False):
    #this is for lstm
    dwi_time_distributed_input = Input(batch_shape = (
                             batch_size, 
                             look_back, 
                             image_size[0], 
                             image_size[1],
                             3
                             ), 
                name="DWI_Input")
    #load pretrained model
    dwi_vgg = make_vgg((image_size[0], 
                        image_size[1],
                        3),
                        "dwi")
    #untrainable pretrained model
    for i,layer in enumerate(dwi_vgg.layers):
        layer.trainable = False
    #this is for lstm
    flair_time_distributed_input = Input(batch_shape = (batch_size, 
                             look_back, 
                             image_size[0], 
                             image_size[1],
                             3
                             ), 
                name="FLAIR_Input")
    #load pretrained model
    flair_vgg = make_vgg((image_size[0], 
                        image_size[1],
                        3),
                     "flair") 

    #untrainable pretrained model
    for i,layer in enumerate(flair_vgg.layers):
        layer.trainable = False
    ##combining pretrained vgg with time distibuted sequences
    dwi_encoded_frames = TimeDistributed(dwi_vgg)(dwi_time_distributed_input)
    flair_encoded_frames = TimeDistributed(flair_vgg)(flair_time_distributed_input)
    
    concat_flair_dwi = Concatenate()([dwi_encoded_frames, flair_encoded_frames])
    
    hidden_lstm = LSTM(64,stateful=True)(concat_flair_dwi)
#    hidden_layer = TimeDistributed(Dense(output_dim=1, activation="relu"))(hidden_lstm)
    outputs = Dense(1, activation="sigmoid")(hidden_lstm)
       
    model = Model(inputs=[dwi_time_distributed_input,
                          flair_time_distributed_input], outputs=[outputs])
    
    optimizer = Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)
    
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]) 
    return model



def make_LSTM_model_singleInput(
                                batch_size =1,
                                sequence_length=10,
                                image_size=(512, 512), 
                                debug=False
                                ):
    #this is for lstm
    dwi_time_distributed_input = Input(batch_shape = (
                             batch_size, 
                             sequence_length, 
                             image_size[0], 
                             image_size[1],
                             3
                             ), 
                name="DWI_Input")
    #load pretrained model
    #load pretrained model
    vgg = make_vgg((image_size[0], 
                        image_size[1],
                        3),
                        "dwi") 
    #untrainable pretrained model
    for i,layer in enumerate(vgg.layers):
        layer.trainable = False

    ##combining pretrained vgg with time distibuted sequences
    dwi_encoded_frames = TimeDistributed(vgg)(dwi_time_distributed_input)

    hidden_lstm = LSTM(256,stateful=True)(dwi_encoded_frames)
#    hidden_layer = TimeDistributed(Dense(output_dim=1, activation="relu"))(hidden_lstm)
    outputs = Dense(1, activation="sigmoid")(hidden_lstm)
       
    model = Model(inputs=[dwi_time_distributed_input], outputs=[outputs])
    
    optimizer = Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]) 
    return model