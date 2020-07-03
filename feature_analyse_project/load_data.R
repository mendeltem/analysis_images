library(feather)
library(pryr)
library(Hmisc)


get_col <- function(df, filter) {
  
  filtered_colnames = list()
  
  idx = list()
  
  for(id in 1: length(colnames(df))) {
    
    if (grepl(filter, colnames(df)[id])){
      filtered_colnames = c(filtered_colnames, colnames(df)[id])
    }
    
  }
  colnames(df_temp)[1:31]
  
  
  filtered_colnames_c = as.character( filtered_colnames)
  
  #c(colnames(df)[1:32],filtered_colnames_c)
  
  return( df_temp[,c(colnames(df)[1:32],filtered_colnames_c)])
}

load_file <- function(path,filtered_colname,limit = 1, 
                            filter1 ="",filter2="",filter3 = "" ) {
  
  all_paths = list.files(path,full.names = TRUE,  recursive = T)
  
  collected_df = data.frame()
  
  filteredlist = list()
  
  if (!is.null(filter1)){
    
    for(i in 1: length(all_paths)) {
      
      if (grepl(filter1, all_paths[i]) &&
          grepl(filter2, all_paths[i]) &&
          grepl(filter3, all_paths[i])
      ){
        
        filteredlist = c(filteredlist, all_paths[i])
      }
    }
  }
  
  for(i in 1: length(filteredlist)) {
    
    #if ( (i %% 5) == 0){
    #  cat("\014")  
    #}
    
    # i-th element of `u1` squared into `i`-th position of `usq`
    print(paste(i, as.character(filteredlist[i]) ))
    print(paste("number of files to load"  ,  length(filteredlist)))
    print(paste("Loaded ", round(i/length(filteredlist)  * 100, digits = 1), "%"))


    if(limit * 10 ^ 9 <= object_size(collected_df)){
      return (collected_df)
    }

    df_temp <- read_feather(as.character(filteredlist[i])) 
    
    df_temp = get_col(df_temp, filtered_colname)
    
    collected_df =rbind(collected_df, df_temp)

    cat(("Loaded Size:"))
    print(object_size(collected_df))
    
    cat("Dimension: \n")
    print(dim(collected_df))

  }
  return (collected_df)
}

# parameter:
# 1: the path
# 2: which Layers? ^1_ in regular expression 
#^1, ^2,...^11  Layers
# 3: limit the size to GB
# 4: experiment filter
#4 examples
#grayscale
#color
#5
#per_hp,per_hp,per_lp,zen_hp,zen_lp,original
#6
#normal, filtered
library(dplyr)
memory_path = '/mnt/data/DATA/dataset/FEATURES/saved/inception/memory'
#search_path = '/mnt/data/DATA/dataset/FEATURES/saved/inception/search'

data = load_file(memory_path,
                 '^11_|^10_|^9_', 
                 8,
                 'color',
                 '',
                 'normal'
                 ) 

head(data)
