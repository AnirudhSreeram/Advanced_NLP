
# Advanced NLP projects

This repo consists of basic ML algorithms from scratch such as naive bayes, multi class classification using perceptrons and linear layer models performed on 4 different datasets. 
The dataset consists of multi class and has different language text data.




## Dataset and Features

Experiments were performed on 4 text datasets which are as follows 
- 4dim
- Questions
- Products
- Odiya

For the features, 4 different features were extracted and experimented
- Bag of Words
- TF-IDF
- Word embeddings from GloVe (50 dim) 
- Word embeddings from fasttext (300 dim)

## ML Algorithms

ML algorithms was written from scratch by using only python and numpy
- Naive Bayes model
- Perceptron model
- single linear model
- multi-layer linear model

Experiments were performed for each of the models on all 4 datasets.


## Demo

To run assignment-1 for classification

```bash
python Assignment-1/classify.py -m <model name> -i <input file> -o <outputfile>
```

To run assignment-2 for classification

```bash 
bash Assignment/run.sh
```
