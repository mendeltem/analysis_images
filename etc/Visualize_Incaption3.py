import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import Sequence
from pylab import rcParams

DEFAULT_IMAGE_DIRECTORY     = 'images/color/session1_memory/original'


file_pictures = "finalresulttab_funcsac_SFC_memory.dat"
all_data = pd.read_table(file_pictures,encoding = "ISO-8859-1")

#Foveal Area
fovea = 30

def data_selecting(data,color,masktype,maskregion,fixincalid = 0,fovea = 30,group = 0, start = 0, stop = 0):
    """choose the data associated to different experiment settings

    Arguments:
        data: DataFram that get filtered
        Experiment type Filter
        colorimages: 1 oder 0 (color oder grayscale images)
        masktype: 0, 1, oder 2 (control, low-pass oder high-pass filter)
        maskregion: 0, 1 oder 2 (control, periphery oder center) 
        
        Daraus ergibt sich entsprechend:
        masktype == 0 & maskregion == 0: Kontrollbedingung
        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
        masktype == 2 & maskregion == 2: zentraler Hochpassfilter
        
        fixinvalid: 0 oder 1 (Fixation ist valide oder invalide); 
        invalide Fixationen sind 
        z.B. blinks oder Fixationen außerhalb des Monitors/Bildes
            
        group:  0:image and subjec 
                1: image 
                2:subject
        head: first indizes of the group

            ...
    Returns: DataFrame with lists of Eyemovement
    """
    
    list_of_ey_xy = pd.DataFrame()
    list_of_ey_id = pd.DataFrame()
    

    #masktype == 0 & maskregion == 0: Kontrollbedingung
    cleaned_data = data.loc[(data["colorimages"] == color) &
                         (data["masktype"]    == masktype) &
                         (data["maskregion"]  == maskregion) ,
                         ['subject',
                          'fixposx',
                          "fixno",
                          "fixposy",
                          "imageid",
                          "masktype",
                          "maskregion",
                          "fixinvalid",
                          "colorimages"]]
    
    #take the first indexes of the group
    #if(head):
        #cleaned_data = cleaned_data.groupby(["imageid","subject"]).head(head)
        
    if(start or stop and group==0):    
        cleaned_data = cleaned_data.groupby(["imageid","subject"]).apply(lambda x: x[start:stop])  
    elif(start or stop and group==1):    
        cleaned_data = cleaned_data.groupby(["imageid"]).apply(lambda x: x[start:stop])    
        
    print(cleaned_data)
    #remove outliers
    cleaned_data = cleaned_data.loc[
                      (cleaned_data["fixposx"] >= fovea) &
                      (cleaned_data["fixposx"] <= 1024 - fovea) &
                      (cleaned_data["fixposy"] >= fovea) &
                      (cleaned_data["fixposy"] <= 768 - fovea)&
                      (cleaned_data["fixinvalid"]  == fixincalid) 
                      ]
    
    
    #print(cleaned_data.loc[:10,"fixno"])
    #debug
    #print(cleaned_data.loc[:,"fixno"])
    #print(cleaned_data)
    
    ###create list of eyemovements
    list_of_ey_id = cleaned_data.groupby("imageid")["imageid"].apply(np.unique)
    list_of_ey_x = cleaned_data.groupby("imageid")["fixposx"].apply(list)   
    list_of_ey_y = cleaned_data.groupby("imageid")["fixposy"].apply(list)
    list_of_ey_xy = pd.concat([list_of_ey_id,list_of_ey_x,list_of_ey_y], axis = 1)
    

    
    return list_of_ey_xy



#masktype == 0 & maskregion == 0: Kontrollbedingung
#group: 0:image and subjec 1: image 2:subject
exp_control_color = data_selecting(all_data,1,0,0,0,group = 0,start=0,stop=1 )



all_meanlist = []
all_pics     = []
#get a set of foveal areas (fixations from Experiment) from a picture 
for imageid in exp_control_color.loc[:,"imageid"].apply(np.unique).iloc[:]:
    pic = plt.imread("images/color/session1_memory/original/" + str(imageid[0]) + ".png")

    meanlist = []

    sel_df = exp_control_color.loc[(exp_control_color["imageid"] == imageid[0]),]
    n_fix = len(exp_control_color.loc[imageid[0],"fixposx"])
    for j in range(n_fix):   
        d = pic[ int(sel_df["fixposy"].iloc[0][j]) - fovea:
                 int(sel_df["fixposy"].iloc[0][j]) + fovea,
                 int(sel_df["fixposx"].iloc[0][j]) - fovea:
                 int(sel_df["fixposx"].iloc[0][j]) + fovea]
        if(d.shape == (60,60,3)):
            meanlist.append(d)
        #print(n_fix)
    all_pics.append(pic)        
    all_meanlist.append(np.array(meanlist))


#debug
#n_fix = len(exp_control_color.loc[5,"fixposx"])
#n_fix


#picture plot size
plt.rcParams["figure.figsize"] = (80,40)
#Image ID
im = 40


plt.subplot(711)
plt.imshow(all_pics[im])
plt.title("Original Picture")
plt.subplot(712)
plt.imshow(np.mean(all_meanlist[im], axis=0))
plt.title("All Mean Fixation Foveal Area")
plt.subplot(713)

plt.imshow(np.mean(all_meanlist[im][:1], axis=0))
plt.title("First Fixation Area")
plt.subplot(714)
plt.imshow(np.mean(all_meanlist[im][-1:], axis=0))
plt.title("LastFixation Area")
plt.subplot(715)
plt.imshow(np.mean(all_meanlist[im][:20], axis=0))
plt.title("First 20 Selected Fixation Area")
plt.subplot(716)
plt.imshow(np.mean(all_meanlist[im][-20:], axis=0))
plt.title("Last 20 Selected Fixation Area")


#plt.imshow(np.mean(all_meanlist[im][20:21], axis=0))






#my Experiment


























