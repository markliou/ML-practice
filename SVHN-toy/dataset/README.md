# SVHN dataset
http://ufldl.stanford.edu/housenumbers/

# install cv2
pup3 install opencv-python

# Process
1. download the SVHN using script
```shell
./downloadSVHN.sh
```
This will download the train, test and extra dataset.

2. build the training dataset list and its lable file.
The label file can be directly copy from the folder of the digitStruct.csv generated from svhn_dataextract_tocsv_nobox.py
The training dataset list are recommended to use the absoluted path.

