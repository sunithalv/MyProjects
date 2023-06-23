# Image Caption Generator
The dataset used is https://www.kaggle.com/datasets/ming666/flicker8k-dataset containing 8k+ images and corresponding captions.
Of these 6k+ images were used for training the model.
1) The model comprises of a Xception model which is pretrained on the imagenet dataset to extract the image features 
2) LSTM network  to get the textual information from the tokenized sequence input
3) The outputs from the above two networks are provided to the dense decoder network to generate the captions
