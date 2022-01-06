
for data in odia 4dim questions products
do
	for batch in 16 32 64 128
	do
		if [ $data == "odia" ] || [ $data == "4dim" ]
		then 
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/fasttext.wiki.300d.vec" -f 10 -l 0.001 -u 1500 -b ${batch} -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/pytorch/relu_softmax/batchs/$data/" -act "softmax" -hact "relu"
		else
			python3 train_pytorch.py -i "datasets/$data.train.txt" -E "datasets/fasttext.wiki.300d.vec" -f 10 -l 0.01 -u 1500 -b ${batch} -e 30 -o  "/Users/anirudhsreeram/Documents/Course-Work/CSCI662/Assignment-2/models/pytorch/relu_softmax/batchs/$data/" -act "softmax" -hact "relu"
		fi

	done
done


