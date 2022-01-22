#!/bin/bash

# This script will automatically download the needed data for the Kaggle challenge

wget -O input.zip https://www.dropbox.com/sh/fhfjjtk0sr7pmse/AAD4ZEtHv9OI5HfVO22tdMX0a?dl=0 
mkdir -p input
unzip input.zip -d input