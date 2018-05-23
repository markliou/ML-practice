#!/bin/bash

# down load dataset
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
wget http://ufldl.stanford.edu/housenumbers/test.tar.gz
wget http://ufldl.stanford.edu/housenumbers/extra.tar.gz

# unzip dataset
tar -zxvf train.tar.gz
tar -zxvf test.tar.gz
tar -zxvf extra.tar.gz

# convert file to jason
#wget https://raw.githubusercontent.com/Bartzi/stn-ocr/master/datasets/svhn/svhn_dataextract_tojson.py

sudo pip3 install h5py
sudo pip3 install optparse
sudo pip3 install json

echo ''
echo ''
echo 'convertin the training mat file to jason...'
python3 svhn_dataextract_tocsv_nobox.py -i train/digitStruct.mat -o train/digitStruct &
echo ''
echo ''
echo 'convertin the test mat file to jason...'
python3 svhn_dataextract_tocsv_nobox.py -i test/digitStruct.mat -o test/digitStruct &
echo ''
echo ''
echo 'convertin the extra mat file to jason...'
python3 svhn_dataextract_tocsv_nobox.py -i extra/digitStruct.mat -o extra/digitStruct 

# make the training dataset list
datasets=( "extra" "test" "train")
for dataset in ${datasets[@]}
do
    # make the list
    for f in `ls ${dataset}/*.png`
    do
        echo `pwd`/${f} >> ${dataset}.lists
    done

    # make the labels
    cp ${dataset}/digitStruct.csv ./${dataset}.labels
done

