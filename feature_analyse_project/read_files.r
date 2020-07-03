setwd('~/Documents/work/mendel_exchange/feature_analyse_project')

library(feather)
library(dplyr)
library(tidyverse)
library(reshape)

path = '/mnt/data/DATA/dataset/FEATURES/saved/exception_mean.file'

df <- read_feather(path)

df = df %>%  arrange(Subject, Image_ID ) 


memory_df = df[grep("memory", df$Experimenttype) ,c(1,2,4,5,seq(from=8, to=19))] 

search_df = df[grep("search", df$Experimenttype) ,c(1,2,seq(from=4, to=19))] 


subject_image_df = memory_df %>% filter( (Subject == 50)  & (Image_ID == 2))


