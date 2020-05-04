# Respiratory-Sound-Data-Classification
CSCE 666 Final Project on Respiratory Sound Data

## Overview

The analysis of lung sounds, collected through auscultation is a fundamental component
in pulmonary disease diagnostics of primary care and general patient monitoring in
telemedicine. Despite advances in computation and algorithms, the goal of automated
lung sound identification has remained elusive. For over 40 years published work in this
field has only had limited success in identifying lung sounds. In this paper report, we
present pattern recognition methods to classify respiratory sounds into a binary
classification of “adventitious” or “not adventitious” on the ICBHI’17 scientific
challenge respiratory sound database. We are also interested in the binary disease
classification problem of the presence/absence of COPD from the analysis of 126
patients' audio responses. We have extracted several features such as MFCC’s, DbMel,
Short Term Fourier Transform(STFT), Chroma Vectors, LPC and Aggregated features
from the audio responses in the database. The extracted features were fed to the three
classifiers namely; SVM(radial basis function), XGBoost and Artificial/Convolutional
Neural Network models. When evaluated on an individual level, time window based
frequency domain features like MFCC’s and STFT performed best on adventitious sound
classification. The classification accuracy for SVM was found out to be 84.34% using the
combination of DB Scaled Mel Features and MFCC’s . For COPD disease classification,
MFCC’s and demographic features were mainly used. The SVM model with
Demographic+MFCC features gave classification accuracy of 98.54%.
