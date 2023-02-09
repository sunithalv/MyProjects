# Malaria-Detection
  This Dataset is taken from the official NIH Website: https://ceb.nlm.nih.gov/repositories/malaria-datasets/ . The dataset has 2 labels parasitized and uninfected with the corresponding cell images. There are a total of 27,558 images.
   
## Model Building
Transfer learning was ised in model building as CNN did not give a good accuracy. vgg19 model was used as it gave the best results.80% of images were used in training   and 20% in testing.
 The model has a train accuracy of 95% and validation accuracy of 90.5%
 
 ## Deployment
 The web app was created using the Flask framework as shown.

![image](https://user-images.githubusercontent.com/28974154/217737202-a318ab67-4431-4a0b-9091-6194f8bffb64.png)
