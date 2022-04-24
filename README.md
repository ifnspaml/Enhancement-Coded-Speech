# Enhancement-Coded-Speech

Please find here the scripts referring to the paper [Convolutional Neural Networks to Enhance Coded Speech](https://arxiv.org/abs/1806.09411). In this repository we provide the cepstral domain approach with the framework structure III. 

The code was written by Ziyue Zhao and Huijun Liu. 

## LATEST 
Some Python code is updated to match the TensorFlow 2 (the original code was written for TensorFlow 1). See Prerequisites for detailed information about how to start. 

## Introduction

An approach based on a convolutional neural network (CNN) is proposed to enhance coded (i.e., encoded and decoded) speech by utilizing cepstral domain features. The quality of coded speech can be enhanced and thus achieves improved quality without modifing the codec (i.e., encoder and decoder) itself.

## Prerequisites and Installation

 - Nvidia GPU with CUDA and CuDNN (the code is tested with CUDA version 11.4)
 - Install [Anaconda](https://www.anaconda.com/)
 - Start Anaconda Prompt
 - Create a new environment and activate: `conda create -n tf-gpu-new python=3.8.5`, `conda activate tf-gpu-new`
 - Install TensorFlow-GPU and Scipy:`pip install -r tensorflow-gpu==2.4.1`, `pip install -r scipy`
 - Install [Matlab](https://www.mathworks.com/) (the code is tested with MATLAB 2016 and later)

## Getting Started

### Testing with the provided CNN model

- Two example files: example_s1_g711_coded.raw and example_s2_g711_coded.raw (the original speech samples are from the [ITU-T test signals](https://www.itu.int/net/itu-t/sigdb/genaudio/AudioForm-g.aspx?val=1000050) of American English) for the G.711-coded speech are included in the `dataset` folder
- Please note that the two example files are split from the file named A_eng_f5.wav in the ITU-T test signals dataset and the splitting point is at 7.0812 s.
- Run the Matlab script to prepare the input data for the CNN model, with G.711-coded speech sample `./dataset/exapmle_s_g711_coded.raw` and the means and standard variances from the training data `./data/mean_std_of_TrainData_g711_best.mat`, outputting the CNN input data `./data/type_3_cnn_input_ceps_v73.mat`, residual cepstral coefficients `./data/type_3_ceps_resi.mat`, and the phase angel vector `./data/type_3_pha_ang.mat`:
```bash
matlab Test_InputPrepare.m
```
- Run the Python script to use the CNN model, with the CNN input data `./data/type_3_cnn_input_ceps_v73.mat` and the provided CNN model `./data/cnn_weights_ceps_g711_best.h5`, outputting the CNN output data `./data/type_3_cnn_output_ceps.mat`:
```bash
python CepsDomCNN_Test.py
```
- Run the Matlab script to obtain the final enhanced speech, with the CNN output data `./data/type_3_cnn_output_ceps.mat`, residual cepstral coefficients `./data/type_3_ceps_resi.mat`, the phase angel vector `./data/type_3_pha_ang.mat`, and G.711-coded speech sample `./dataset/exapmle_s_g711_coded.raw`, outputting the enhanced speech waveform `./dataset/example_s1_g711_coded_cnn_proc.raw` or `./dataset/example_s2_g711_coded_cnn_proc.raw`:
```bash
matlab Test_WaveformRecons.m
```
### Reproduce the results

The results reported in the paper is tested on the NTT wideband speech database, so if you want to reproduce the exact results, the test need to be done with the same speech data (see details in the paper). 

### Training with your own dataset

- Run the Matlab script to prepare the CNN training data, with the uncoded speech for training `./dataset/example_uncoded_train_s.raw`, uncoded speech for validation `./dataset/example_uncoded_valid_s.raw`, coded speech for training `./dataset/example_coded_train_s.raw`, and coded speech for validation `./dataset/example_coded_valid_s.raw`, outputting training input `./data/Train_inputSet_g711.mat`, training target `./data/Train_targetSet_g711.mat`,  validation input `./data/Validation_inputSet_g711.mat`, validation target `./data/Validation_targetSet_g711.mat`, and the means and standard variances from the training data `./data/mean_std_of_TrainData_g711_example.mat`:
```bash
matlab Training_Data.m
```
- Run the Python scripts to train the CNN model, with the above-mentioned CNN training data, outputting the trained CNN weights `./data/cnn_weights_ceps_g711_example.h5`:
```bash
python CepsDomCNN_Train.py
```
- Note that your own dataset needs to replace the example speech files (the example speech samples are from the [ITU-T test signals](https://www.itu.int/net/itu-t/sigdb/genaudio/AudioForm-g.aspx?val=1000050) of American English).

### Codecs and processing functions

- To obtain G.711-coded speech samples, some processing functions and the ITU-T G.711 codec are needed.
- Download the processing functions from [ITU-T G.191](https://www.itu.int/rec/T-REC-G.191-201003-I) and compile the relevant files to obtain the programs: `filter.exe`, `sv56demo.exe`, and `g711demo.exe`.
- Put the compiled programs in the root directory.
- Run the Matlab script to obtain G.711-coded speech, with a raw speech sample `./dataset/exapmle_s.raw` and the above-mentioned programs, outputting G.711-coded speech `./dataset/exapmle_s_g711_coded.raw`:
```bash
matlab CodedSpeech_Obtain.m.
```

## Citation

If you use the scripts in your research, please cite

```
@article{zhao2019convolutional,
  author = {Z. Zhao and H. Liu and T. Fingscheidt},
  title = {{Convolutional Neural Networks to Enhance Coded Speech}},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year = {2019},
  month = april,
  volume = {27}, 
  number = {4},
  pages = {663-678}
}
@article{cnn2codedspeech,
  author =  {Z. Zhao and H. Liu and T. Fingscheidt},
  title =   {{Convolutional Neural Networks to Enhance Coded Speech}},
  howpublished = {\url{https://github.com/ifnspaml/Enhancement-Coded-Speech}},
  year =    {2018},
  month =   jun
}
```

## Acknowledgements
- The CNN topology used here is a deep encoder-decoder network, which is motivated from [Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections](https://arxiv.org/abs/1603.09056).
- The authors would like to thank Samy Elshamy, Jonas Löhdefink, and Jan-Aike Bolte for the advice concerning the construction of the project in Github.
