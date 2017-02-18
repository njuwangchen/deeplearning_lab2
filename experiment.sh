#! /bin/bash

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 100 0.01 0.9 0 RELU
python python/early_curve.py python/data/100_0.01_0.90_0.00_RELU

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 100 0.01 0.9 0 Sigmoid
python python/early_curve.py python/data/100_0.01_0.90_0.00_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 1000 0.01 0.9 0 RELU
python python/early_curve.py python/data/1000_0.01_0.90_0.00_RELU

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 1000 0.01 0.9 0 Sigmoid
python python/early_curve.py python/data/1000_0.01_0.90_0.00_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.1 0.9 0 Sigmoid
python python/early_curve.py python/data/10_0.10_0.90_0.00_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.25 0.9 0 Sigmoid
python python/early_curve.py python/data/10_0.25_0.90_0.00_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.0 0 Sigmoid
python python/early_curve.py python/data/10_0.01_0.00_0.00_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.0 0.01 Sigmoid
python python/early_curve.py python/data/10_0.01_0.00_0.01_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.0 0.05 Sigmoid
python python/early_curve.py python/data/10_0.01_0.00_0.05_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.0 0.1 Sigmoid
python python/early_curve.py python/data/10_0.01_0.00_0.10_Sigmoid

java -classpath out/production/deeplearning_lab2 Lab2 data/protein-secondary-structure.txt 10 0.01 0.9 0.05 Sigmoid
python python/early_curve.py python/data/10_0.01_0.90_0.05_Sigmoid