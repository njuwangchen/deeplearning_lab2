#! /bin/bash

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.9 0 RELU
python python/early_curve.py python/data/10_0.01_0.90_0.00_RELU