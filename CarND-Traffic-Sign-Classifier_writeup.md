# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./additional_signs/1.jpg "Class type 1"
[image5]: ./additional_signs/4_b.png "Class type 4"
[image6]: ./additional_signs/12.PNG "Class type 12"
[image7]: ./additional_signs/13_b.jpg "Class type 13"
[image8]: ./additional_signs/14.jpg "Class type 14"
[image60]: ./additional_signs/18_c.png "Class type 18"
[image70]: ./additional_signs/23.jpg "Class type 23"
[image80]: ./additional_signs/25.jpg "Class type 25"
[image90]: ./additional_signs/28_b.png "Class type 28"
[image100]: ./additional_signs/38_b.png "Class type 38"
[image11]: ./examples/color_vs_gray.png "Image before and after applying grayscaling.."
[image12]: ./examples/Original_images.png "Original images"
[image9]: ./examples/image_count_vs_sign_type.png "Number of images vs Class type"
[image10]: ./examples/table_image.png "Number of images per Class type"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Thanks for reviewing the project, here is a link to the project Jupyter notebook [project code](https://github.com/proshan75/-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The histogram shown below displays number of images per class ID. This bar chart provides the overview and indicates that some of the class types have large number of images while others have significantly less. The impact of few images will be observed in learning and predicting those types of images.

![alt text][image9]

Though the histogram shows overall distribution, I wanted to know specific number of images per class types. So here I generated a table plot that shows number of images per class type. Following table image shows those specifics:

![alt text][image10]

Finally, I plotted one image for each class type. As there are multiple images for each class type, here I displayed the class specific image with random selection. This view gives the most valuable visual information. It shows some images are nice and clean while others are very dark and hard to see. Following image shows few of the origginal images:

![alt text][image12]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the noise in the image and shows the content in terms of brighness. This in turn helps to identify and distinguish the visual features, resulting in better prediction. Also, it improves the computational aspect.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image11]

As a last step, images are normalized. The visualization of the images shows varying range of image brighness. By normalizing all the images, the image data is uniformally represented in the range on 0 to 1.  

Due to time constraint and delay in delivering the project (for various reasons), I limited the processing steps to grayscaling and normalization. 
I hope to get back to the project after submission (and once I get caught up with the class schedule) I plan to augment the original image data set by implementing rotation, scaling etc.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x3 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 24x24x6 	|
| Softsign				|												|
| Max pooling		   	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling		   	| 2x2 stride,  outputs 5x5x6 					|
| Flatten			    | Output 400   									|
| Fully connected		| Input 400 Output 120							|
| RELU					| 												|
| Fully connected		| Input 120 Output 84							|
| RELU					| 												|
| Fully connected		| Input 84 Output 43							|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer same in the original example. However I played a lot with the hyper parameters to get the accuracy above 0.93. I settled on the following values as they provided accuracy 0.956.
* Batch size: 96
* Number of epochs: 100
* Learning rate: 0.0005


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.956 
* test set accuracy of 0.934

I initially started with LeNet architecture used for training and predicting numbers. After trying out many iterations on the hyperparameters I was barely getting validation accuracy around 0.87. At that point I decided to enhance the preprocessing to include grayscale image step. That improved the accuracy a bit. Further I enhanced the preprocessing to crop the images which allowed the accuracy around 0.93. However it was not consistent and sometime I noticed getting below 0.93.

At point I decided to change the architecture and include one additional convolution layer. Also, I used softsign activation function. That helped me to improve the validation accurary to consistently above 0.95. 

Among the hyperparameters, the number of epoch seems to be the one affecting the accuracy the most. If its value is low then the accuracy doesn't reach to the threshold. Learning rate of 0.0003 or 0.0005 worked on the best to get the high accuracy. Lowering further seems to affect adversly on the accuracy. Similarly the batch size helped to improve the accuracy in the range from 64 to 128, choosing 96 gave accuracy above 0.95.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image60] 
![alt text][image70] ![alt text][image80] 
![alt text][image90] ![alt text][image100]  

The first image (class type 1) might be difficult to classify as it has some watermark.

The second image (class type 4) should be predicable accurately as the sample size for this class type is fairly high.

The third image (class type 12) should be predicable accurately as the sample size for this class type is fairly high.

The fourth image (class type 13) should be predicable accurately as the sample size for this class type is fairly high.

The fifth image (class type 14) might be difficult to classify as it is a bit skewed and relatively lower number of images for this class type.

The sixth image (class type 18) might be difficult due to a watermark on the image, also there is additional text board below the actual sign.

The seventh image (class type 23) might be difficult to many details in the image for indicating skidding vehicle. Another factor is the sample size of this class type, the number of images seems a bit lower that others.

The eigth image (class type 25) might be difficult to many details in the image for indicating vehicle.

The ninth image (class type 28) might be difficult due to details in the image and relatively smaller sample size of images.

The tenth image (class type 38) should be predicable accurately as the sample size for this class type is fairly high.




#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h       		| 30 km/h   									| 
| 70 km/h       		| 30 km/h   									| 
| Priority road 		| Priority road									| 
| Yield         		| Yield     									| 
| Stop Sign     		| Stop sign   									| 
| General Caution		| General Caution								|
| Slippery road			| Bicycles crossing								|
| Road work	      		| Bicycles crossing				 				|
| Children crossing		| Beware of ice/snow   							|
| Keep right			| Keep right        							|


The model was able to correctly guess 6 of the 10 traffic signs, which gives an accuracy of 60%. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 30 km/h sign (probability of 1.0), and the image does contain a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 30 km/h   									| 
| .0     				| 50 km/h 										|
| .0					| 80 km/h										|
| .0	      			| Wild animals crossing			 				|
| .0				    | Stop              							|


For the second image, the model is relatively sure that this is a 70 km/h sign (probability of 0.997), and however the image  contains a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.997        			| 30 km/h   									| 
| .001     				| Turn right ahead								|
| .001					| 60 km/h										|
| .001	      			| Roundabout mandatory			 				|
| .0				    | Children crossing    							|


For the third image, the model is relatively sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road									| 
| .0     				| Roundabout mandatory							|
| .0					| No vehicles									|
| .0	      			| Keep right        			 				|
| .0				    | End of all speed and passing limits			|


For the fourth image, the model is relatively sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield     									| 
| .0     				| Keep right        							|
| .0					| Ahead only									|
| .0	      			| Children crossing    			 				|
| .0				    | Turn left ahead   							|


For the fifth image, the model is relatively sure that this is a Stop sign (probability of 1.0), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop      									| 
| .0     				| Turn left ahead   							|
| .0					| 20km/h    									|
| .0	      			| Traffic signals      			 				|
| .0				    | No vehicles 									|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


