library(feather)
library(dplyr)

exception_path = '/mnt/data/DATA/dataset/FEATURES/saved/exception_mean.file'

inception_path = '/mnt/data/DATA/dataset/FEATURES/saved_2/inception_mean.file'

#lade den DataFrame 
inception_df <- read_feather(inception_path)
exception_df <- read_feather(exception_path)
#sortiere nach Proband_id und Bild_id

exception_df = exception_df %>%  arrange(Subject, Image_ID ) 
inception_df = inception_df %>%  arrange(Subject, Image_ID ) 

exception_memory_df = exception_df[grep("memory", exception_df$Experimenttype) ,c(2,4,5,seq(from=8, to=19))] 
exception_search_df = exception_df[grep("search", exception_df$Experimenttype) ,c(2,seq(from=4, to=19))] 

inception_memory_df = inception_df[grep("memory", inception_df$Experimenttype) ,c(2,4,5,seq(from=8, to=19))] 
inception_search_df = inception_df[grep("search", inception_df$Experimenttype) ,c(2,seq(from=4, to=19))] 

write_feather(exception_memory_df, '/mnt/data/DATA/dataset/FEATURES/saved/exception_memory_mean.file')
write_feather(exception_search_df, '/mnt/data/DATA/dataset/FEATURES/saved/exception_search_mean.file')
write_feather(inception_memory_df, '/mnt/data/DATA/dataset/FEATURES/saved/inception_memory_mean.file')
write_feather(inception_search_df, '/mnt/data/DATA/dataset/FEATURES/saved/inception_search_mean.file')

#summary(exception_memory_df)
#subject_image_df = inception_search_df %>% filter( (Subject == 50)  & (Image_ID == 2))

#features von exception model
per_lp_exception_memory_df = exception_memory_df %>% filter(Filtertype == 'per_lp')
zen_lp_exception_memory_df = exception_memory_df %>% filter(Filtertype == 'zen_lp')
per_hp_exception_memory_df = exception_memory_df %>% filter(Filtertype == 'per_hp')
zen_hp_exception_memory_df = exception_memory_df %>% filter(Filtertype == 'zen_hp')
original_exception_memory_df = exception_memory_df %>% filter(Filtertype == 'original')

per_lp_exception_search_df = exception_search_df %>% filter(Filtertype == 'per_lp')
zen_lp_exception_search_df = exception_search_df %>% filter(Filtertype == 'zen_lp')
per_hp_exception_search_df = exception_search_df %>% filter(Filtertype == 'per_hp')
zen_hp_exception_search_df = exception_search_df %>% filter(Filtertype == 'zen_hp')
original_exception_search_df = exception_search_df %>% filter(Filtertype == 'original')

#features von inception model
per_lp_inception_memory_df = inception_memory_df %>% filter(Filtertype == 'per_lp')
zen_lp_inception_memory_df = inception_memory_df %>% filter(Filtertype == 'zen_lp')
per_hp_inception_memory_df = inception_memory_df %>% filter(Filtertype == 'per_hp')
zen_hp_inception_memory_df = inception_memory_df %>% filter(Filtertype == 'zen_hp')
original_inception_memory_df = inception_memory_df %>% filter(Filtertype == 'original')

per_lp_inception_search_df = inception_search_df %>% filter(Filtertype == 'per_lp')
zen_lp_inception_search_df = inception_search_df %>% filter(Filtertype == 'zen_lp')
per_hp_inception_search_df = inception_search_df %>% filter(Filtertype == 'per_hp')
zen_hp_inception_search_df = inception_search_df %>% filter(Filtertype == 'zen_hp')
original_inception_search_df = inception_search_df %>% filter(Filtertype == 'original')



length(exception_df$index)

exception_df[1:length(exception_df$index),c(2,4,5,seq(from=8, to=19))] 






