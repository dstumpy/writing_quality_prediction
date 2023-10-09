#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR

# dowload kaggle dataset using kaggle API
FILE=$SCRIPT_DIR/linking-writing-processes-to-writing-quality.zip
echo $FILE

if [ ! -f "$FILE" ]; 
then
    echo "File missing. Download file using kaggle API."
    kaggle competitions download -c linking-writing-processes-to-writing-quality
else
    echo "File already existing."
fi

# unzip loaded data
echo "Unzip $FILE"
unzip -d $SCRIPT_DIR/raw -o $FILE
