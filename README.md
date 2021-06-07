# Machine Learning


## Description
This repository is for building Machine Learning model and predict the label of a document.


## Requirements
- Python 3.8.8
- Sastrawi 1.0.1
- NLTK 3.6.1
- Bert (bert-for-tf2) 0.14.9
- TensorFlow 2.5.0
- TensorFlow Hub 0.12.0


## Dataset
- Training data from [JDIH Kemenkeu](https://jdih.kemenkeu.go.id/in/home). Downloaded 9176 documents and preprocessed into 8795 documents.
- Prediction data from [JDIHN](https://jdihn.go.id/). Downloaded 55374 documents and preprocessed into 52836 documents.
- There are 54 unique labels.
- The feature used for the model is the title of the documents.


## Problem
A multi-label classification problem which then transformed into a binary classification problem to be solved.


## Model
- Using Binary Cross Entropy for loss.
- Using Adam for optimizer.
- Using [BERT](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2) transfer learning for embedding layer.

## References
- [Multi-label Text Classification with Scikit-learn and Tensorflow](https://medium.com/swlh/multi-label-text-classification-with-scikit-learn-and-tensorflow-257f9ee30536)
- [A detailed case study on Multi-Label Classification with Machine Learning algorithms and predicting movie tags based on plot summaries!](https://medium.com/@saugata.paul1010/a-detailed-case-study-on-multi-label-classification-with-machine-learning-algorithms-and-72031742c9aa)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Simple BERT using TensorFlow 2.0](https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22)
- [How does Keras handle multilabel classification?](https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification/44165755)