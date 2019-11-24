#!/bin/bash
dir=`pwd` 
name=${dir##*/} 
echo "$name" 
find train_data | grep -E "jpg|bmp" | sort | awk -F "/" '{print $0 " " $2 }' > ${name}_train.txt 
echo "Gen ${name}_train.txt success..." 
find test_data | grep -E "jpg|bmp" | sort | awk -F "/" '{print $0 " " $2 }' > ${name}_test.txt 
echo "Gen ${name}_test.txt success..." 
