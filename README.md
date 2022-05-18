# CloudDetection-Utilitycode

# 1.LandSat8 Biome
The data is cut into 512 * 512 patches with overlapping
The data files are arranged as follows：
 LandSat8Biome
  |__512(Patch Size)
    |__B1:patch name.tif
    |__B2
    |__B3
    |__B4
    |__B5
    |__B6
    |__B7
    |__B8
    |__B9
    |__B10
    |__B11
    |__mask:patch name_mask.tif
    |__datatype_patch_name.csv(ContainingFile all patches' names, does not contain suffix)
   |__256(Patch Size)
 
# 2.Sentinel-2 Cloud Mask Catalogue
This dataset comprises cloud masks for 513 1022-by-1022 pixel subscenes, at 20m resolution, sampled random from the 2018 Level-1C Sentinel-2 archive. 
The data format is numpy array file（.npy）
The data is cut into 512 * 512 patches with overlapping
The data files are arranged as follows：
 Sentinel2
  |__NoShadow
    |__img:Patch Name.npy
    |__mask:Patch Name.npy
    |__datatype.csv(ContainingFile all patches' names, including suffix)

# 3.LandSat8Sentinel2Data
This data set simply outputs a pair of sentinel2 and landsat8 data in order to explore the cloud detection method across remote sensing platforms
