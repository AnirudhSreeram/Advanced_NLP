#!/bin/bash
stage=2

source /Users/anirudhsreeram/miniconda3/etc/profile.d/conda.sh
conda activate NLP

if [ $stage -le 1 ]; then
#embed=glove.6B.50d.txt
embed=fasttext.wiki.300d.vec

echo "############## Stage 1 #################"
for data in questions 4dim odia products
do
	for batch in 32 64 128
	do
		for lr in 0.1 0.01 0.001
		do

		python3 train.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f 10 -l $lr -u 300 -b ${batch} -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/myimp/$embed/$data" -act "softmax" -hact "relu"
		python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f 10 -l $lr -u 300 -b ${batch} -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/pytorch/$embed/$data" -act "softmax" -hact "relu"

	 	done
	done
done
fi

if [ $stage -le 2 ]; then
echo "############## Stage 2 #################"
for data in questions 4dim odia products
do
	for maxlen in 10 20
	do
		if [ $data == questions ]
		then
			embed=fasttext.wiki.300d.vec
			python3 train.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 64 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/myimp/" -act "softmax" -hact "relu"
			embed=glove.6B.50d.txt
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 128 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/pytorch/" -act "softmax" -hact "relu"
		elif [ $data == 4dim ] 
		then
			embed=fasttext.wiki.300d.vec
			python3 train.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 64 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/myimp/" -act "softmax" -hact "relu"
			embed=glove.6B.50d.txt
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 64 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/pytorch/" -act "softmax" -hact "relu"
		elif [ $data == odia ]
		then
			embed=fasttext.wiki.300d.vec
			python3 train.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 32 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/myimp/" -act "softmax" -hact "relu"
			embed=fasttext.wiki.300d.vec
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.1 -u 300 -b 32 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/pytorch/" -act "softmax" -hact "relu"
		else
			embed=glove.6B.50d.txt
			python3 train.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.01 -u 300 -b 64 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/myimp/" -act "softmax" -hact "relu"
			embed=glove.6B.50d.txt
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/$embed" -f $maxlen -l 0.01 -u 300 -b 128 -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/finamodels/pytorch/" -act "softmax" -hact "relu"
		fi
	done
done
fi


