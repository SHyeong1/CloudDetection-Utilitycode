# CloudDetection-Utilitycode<br>
## 1.CloudData Set
### 1.1LandSat8 Biome<br>
The data is cut into 512 * 512 patches with overlapping<br>
The data files are arranged as follows：<br>
&ensp;**LandSat8Biome**<br>
&ensp;&ensp;|__512(Patch Size)<br>
&ensp;&ensp;&ensp;|__B1:patch name.tif<br>
&ensp;&ensp;&ensp;|__B2<br>
&ensp;&ensp;&ensp;|__B3<br>
&ensp;&ensp;&ensp;|__B4<br>
&ensp;&ensp;&ensp;|__B5<br>
&ensp;&ensp;&ensp;|__B6<br>
&ensp;&ensp;&ensp;|__B7<br>
&ensp;&ensp;&ensp;|__B8<br>
&ensp;&ensp;&ensp;|__B9<br>
&ensp;&ensp;&ensp;|__B10<br>
&ensp;&ensp;&ensp;|__B11<br>
&ensp;&ensp;&ensp;|__mask:patch name_mask.tif<br>
&ensp;&ensp;&ensp;|__datatype_patch_name.csv(ContainingFile all patches' names, does not contain suffix)<br>
&ensp;&ensp;|__256(Patch Size)<br>
 
### 1.2.Sentinel-2 Cloud Mask Catalogue
This dataset comprises cloud masks for 513 1022-by-1022 pixel subscenes, at 20m resolution, sampled random from the 2018 Level-1C Sentinel-2 archive. <br>
The data format is numpy array file（.npy）<br>
The data is cut into 512 * 512 patches with overlapping<br>
The data files are arranged as follows：<br>
&ensp;**Sentinel2**<br>
&ensp;&ensp; |__NoShadow<br>
&ensp;&ensp;&ensp;|__img:Patch Name.npy<br>
&ensp;&ensp;&ensp;|__mask:Patch Name.npy<br>
&ensp;&ensp;&ensp;|__datatype.csv(ContainingFile all patches' names, including suffix)<br>

### 1.3.LandSat8Sentinel2Data
This data set simply outputs a pair of sentinel2 and landsat8 data in order to explore the cloud detection method across remote sensing platforms
