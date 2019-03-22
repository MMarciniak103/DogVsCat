# DOG VS CAT
This project was made as an academic assignment (That is why comments had to be in polish language). Main goal was to make a functionable and interesting application in Matlab. At the same time I was working on image classification task with usage of deep learning, so I decided to combine both goals and that's how idea for this app was born.

## Prerequisites
It's necessary to have tensorflow and openCV installed. It is also required to get numpy (but I assume that most people have it already installed)

`pip install --upgrade tensorflow`

`pip install opencv-python`

`pip install numpy`

## Description
Deep Convolutional Network was trained in google collab using pets dataset from kaggle: [kaggle-dataset](https://www.kaggle.com/c/dogs-vs-cats)
Interface was made in Matlab and it is communicating with python code.

### You can choose which photo do you want to classify
![1](https://user-images.githubusercontent.com/37276611/54849738-6f4c3d80-4ce5-11e9-8cb9-a89664d4bf65.PNG)

### When you click predict button photo is sent to python script and neural net is making prediction that is sent back to the interface
![3](https://user-images.githubusercontent.com/37276611/54849854-bd614100-4ce5-11e9-9ed4-88af2a620786.PNG)
