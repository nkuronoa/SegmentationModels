#!/bin/sh

INPUT="../../../Datasets/PublicDatasets/CamVid"
MODELTYPE="UNet++"
ENCODER="resnet34"

python main/src/eval.py -i ${INPUT} -mt ${MODELTYPE} -encoder ${ENCODER}