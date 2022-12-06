#!/bin/bash

# go to data directory
cd tmp_data

# declare array with file names
declare -a FileNamesArray=(
    "000__14_15_19" "001__14_19_19" "002__14_23_19" "003__14_27_20" "004__13_10_20" "004__14_31_20"
    "005__14_35_20" "006__11_44_59" "006__14_39_20" "007__11_48_59" "007__13_22_20" "007__14_43_20"
    "008__11_52_59" "008__13_26_20" "008__14_47_20" "009__11_57_00" "009__14_51_20" "010__14_55_20"
    "011__13_38_20" "011__14_59_20" "012__13_42_20" "012__15_03_21" "013__13_46_21" "013__15_07_21"
    "014__13_50_21" "014__15_11_21" "015__13_54_21" "015__15_15_21" "016__15_19_21"
)

# get Thermo Presence data from https://github.com/PUTvision/thermo-presence
for FILENAME in "${FileNamesArray[@]}"
do
    wget -nv -nc https://github.com/PUTvision/thermo-presence/raw/master/dataset/hdfs/$FILENAME.h5
done
