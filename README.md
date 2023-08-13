# Neural Networks
## The Purpose
I always wanted to get into ML projects yet I always questioned: how? This project is my answer. I could just have watched a tutorial and be done with it (which is a valid way to start) but that was not what I wanted to do. I decided to try and learn some of the architecture behind a Neural Network (NN) through this project.

### What I Intend
I intend to learn how I can create a NN from scratch to see how different types of functions, methods, and everything else can work together to train and test on a dataset. I'll get to what I did in the next section which is [here](#The-Project). To help differentiate a bit this project will also compare computational speed differences between languages. NN's and any training they go through does take a while (depending on size) so I want to see: how much does language affect that speed? So that's what I'll look into!

## The Project

### The Dataset
I am using the EMNIST dataset that combines the original MNIST dataset with more digits by adding altered versions of the original. There are about 300000 digit images (originally only 70000) 28 x 28 pixels in size. The MNIST dataset of 70000 feature more simple versions of digits (0 - 9) while the EMNIST dataset features different handwriting styles as well as numbers that are cropped, rotated, and written with errors.

### The Code
The project is made in Python first, C++ and Java second. The Python implementation from scratch is based on code written by [Samson Zhang](https://youtu.be/w8yWXqWQYmU) and the C++ and Java implementations are sort of "translated" versions. The "Without Libraries Section" on the Repo is the code I built upon Zhang's original code. I added time calculation, accuracy and loss calculations (For validation and training sets), as well as plots for those respective elements. All the important information from each NN run is saved into the output folder automatically. The version written with the use of a library is code I wrote from [NeuralNine](https://youtu.be/bte8Er0QhDg).

### Some Results / Findings
