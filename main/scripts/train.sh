#!/bin/sh

INPUT="../../../Datasets/PublicDatasets/CamVid"
MODELTYPE="UNet++"
ENCODER="resnet34"

python main/src/main.py -i ${INPUT} -mt ${MODELTYPE} -e 50 -b 8 -encoder ${ENCODER}