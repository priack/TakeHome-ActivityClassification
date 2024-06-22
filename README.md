# Introduction
This is the code exercise for the Machine Learning Engineer position at Raia. The goal was to create an activity classification model from accelerometer data.

# How to run
Just execute the main.py file.The output should be the confusion matrix present in this document as well. Further exploration of the data can be done manually.

# Data format
For this exercise I used the raw dataset (it had to be manually corrected as it contained several rows with incorret format), because the features selected by default seemed in-adequate to me.

The main problem with the pre-defined features was that they were calculated independently in the x, y, and z axis. The problem with that is that the xyz in the device frame are not constant, e.g. the x axis could correspond to the East direction at t=0 and to the Up direction at t=1. This can easily be confirmed by plotting the three 
axis of the accelerometer and see that none of them is more or less constantly 10 (which is the earths gravity). There are three ways of dealing with this problem:
  1. Using an AHRS to rotate the acceleration form the device frame to the ENU (East, North, Up) frame. However, without having the gyroscope this can be tricky.
  2. Using a classifier that is rotation invariant. This is possible, but increases the complexity of the classifier in a very significant way.
  3. Merge the three dimmensions by, for example, caluting the norm. The problem of this approach is that the direction of the motion is lost. 

I decided to go with the third option for several reasons:
 1. Activity detection is not heavily influenced by the direction of motion. The norm of the accelerometer is going to be the same if you are running in the North direction or the East. However, removing the gravity would be a good idea.
 2. It is the simplest implementation of the three.
 3. Reduces the feature space by 3 (you don't have individual x, y, and z components), which will reduce noise, training and inference time.
 4. In my previous experience is the best approach.

# Data pre-processing
Due to the time-domain aspect of the data, to be able to extract features from the raw data, the first thing to do is to define the windows over which the features are going to be calculated. Due to the fact that there seems to be a variable sample rate (even if it is stated that is 20 Hz) and there seems to be a lot of data droppings (only ~70% of consecutive samples are 45-55 ms appart), a variable window size based on the time stamp was implemented. Other options were explored but I was not convinced of their accuracy. This approach takes long pre-processing time than one that assumes that the sample rate is constant, but produces more trustworthy results. For the stride, a fix number (independent of time) was used. For the stride value, 25 samples were selected. In theory this would be 1.25 s, but it will vary. This reduces the number of samples by a factor of 25, which may seem to be bad, but using a stride of 1 results in a huge overlap between the windows and leads to much longer processing / training times without providing a meaningful increase in the performance. However, this is a value that should be explored to find the propper balance.

For each window, then we calculate 8 features:
  1. Mean
  2. Variance
  3. Skewness
  4. Kurtosis
  5. Min value
  6. Max value
  7. Number of peaks
  8. Mean peak amplitude

For the peak detection, an adaptive peak detection algorithm with moving threshold was used. The advantage of this method is that is independent on the size of the input, so the free paramethers are easier to set. This selection of features is solely based in my previous experience with this problem.

After this process, we end up with 42847 samples. Of these, 10% were used for testing and the remaining 90% for training and validation. However, instead of selecting those samples randomly, the 10% of each subject and activity were selected for testing. This allow us to reduce the overlap between the raw data used for training and testing (there is no overlap in the feature space), and by using the last part we mantain the temporal coherence of the data (i.e. we are not using future data to train). This would be more relevant in a transfer learning approach (not implemented in this case), but it is always nice to have. This split also allow us to ensure a 'representative' test set for every user and activity.

As aditional step in the pipeline, a Thresshold scaler based on Tukey's fences was implemented to clip the outliers. This require a separation of training and testing data, so it was done a posteriori of the feature calcualation.

# Model
Random forest was selected as the classification model due to it's explainability, natural capability of multiclass classification, and lightweight. GradientBoostClassifier was also attempted, but the training time and hyperparameter space were too large. To select the hyperparameters, scikit-learn's GridSearchCV was used. This object allows to explore all the combinations of different hyperparameters in one line. Three hyperparameters were selected for the exploration, each of them with three levels:
  1. Criterion: Gini / Entropy / Log loss. It didn't had almost any effect. So Entropy was selected as it's results were very slightly better.
  2. Number of estimators: 5, 10 ,100. This was the most relevant hyperparameter (no surprise here). The fact that the performance increase wiht the number of estimators (no saturation), indicates that this is a parameter that  would require more exploration.
  3. Max depth: 3, 7, 12. This hyperparameter was also relevant, but not as much as the number of estimators. It didn't show saturation either, but I  would advise agiainst increasing it much, since it could lead to overfitting.

This search was repeated in a 5-fold validation way. This means that each combination of hyperparameter was tested 15 times (3 times for the grid search cross validation, times 5 the fold validation). While this can be an overkill, it ensures the proper selection of the best configuration.

Once the best configuration was selected, a new model was trained using the whole training data.
Finally, this last model was used to predict the values of our left-out data (the testing data). This resulted in a weighted accuracy of 70.7%, a f1-macro score of 0.722 and a Cohen's kappa score of 0.725. All in all a very good result for an inbalance 6 class problem.
Furthermore, if we look at the confusion matrix, we can see that there are clearly 2 blocks of data. The ones that correspond to motion activities, and the ones that correspond to inmobile activities. The biggest missclassification occurs in the inmobile activities (sitting and standing) with 23-24% of confusion between them. Honestly, I am surprise that is this low, since the data from the accelerometer in those two conditions can be extremely similar (since there is no acceleration while inmobile). I would explore more these results to ensure that there is no  some problem there. Other than that, the activity that show the worst performance is Downstairs, which is confused a lot with Walking and, to some extent, with Upstairs. The first error is probably due to the data imbalance, while the second could be a similarity between both classes. Probably features that better identify the vertical motion should be implemented to reduce the error of these two classes. Also, more data would be benefficial.
![alt text](https://github.com/priack/Raia_exercise/blob/main/confusionMatrix.png)


# Limitations
The main limitation in this exercise has been the alloted time (5  hours). I spent more that that, in particular to create this document. This limited the quality of the results and analysis performed, since trying different approaches would have taken too much time.  In the past I spent weeks / months to solve this problem, 5 hours to me seems totally insufficient.

This has lead to the following limitations:
  1. Code quality. I didn't have time to create proper documentation in the code (typing, format, usage etc). My code for deployment always include at least the minimum documentation.
  2. Pre-processing parameters. I used a window of 5 seconds (compared to the 10 s that is used by default) because in my experience this is enough and provides quicker results. However, the performance compared to a 10 s window is probably lower. An analysis on the window size (as well as the stride) would have been an interesting addition to the analysis.
  3. Signal processing. In my experience accelerometer data doesn't need much pre-processing so I ended using just the raw data (no filters). However, I noticed that the sample frequency is quite irregular, so some methods to aleviate that could have been implemented. Also, I didn't notice much noice in a quick visualization of the data, but a deeper analysis could be performed.
  4. Feature exploration. I used the features that I had used in the past, without a proper feature analysis and selection.
  5. Model exploration. At the end I trained only RandomForest, since the GradientBoostClassifier took too much time to train in my machine. Also the hyper parameter values used for exploration were quite limited for the same reasons. This is also the reason for not using Neural Networks.
  6. The final results are just a snapshot of the metrics for the test dataset. Usually I would prefer to have a more detailed analysis, with the distribution of different training models etc.
  7. Report. The report ended up being just this README file, that barely contains any figure demonstrating the points mentioned in it. With more time I would have created a proper Jupyternotebook with all the different steps and figures to support my statements.

# Future work
Before commiting to Random Forest as classification model, I would explore different architectures such as Neural Networks or the above mentioned GRadientboost classifier that has show quite good results. Also, depending on the requiremetns of the deployed solution, I would integrate temporal information into the model. Right now every window is classified independently from the past. Something like a Kalman filter could be used to determine the actual activity based no only on the latest prediction but by the past ones. This would increase the response time, but also the classification accuracy, as spurios events or outliers, would be filtered out.
Another venue to explore would be the use of Neural Networks, in particular recursive ones, to be able to include the time information. Also, using CNN could lead to new features, instead of the hand crafted ones.

# Additions
To keep it easy to track, I will add here the results that I further explored in my own time. I've also increased the quality of code. As remainder, the default values were 5 s and stride 25. 
Confusion matrix for window length 10 s and stride 25
![alt text](https://github.com/priack/Raia_exercise/blob/main/confusionMatrix_10s.png)
The corresponding metris were: Accuracy: 0.732	 F1: 0.740	 Kappa: 0.755

Confusion matrix for window lengths 10 s and stride 25
![alt text](https://github.com/priack/Raia_exercise/blob/main/confusionMatrix_15s.png)
The corresponding metris were: Accuracy: 0.732	 F1: 0.731	 Kappa: 0.760


Confusion matrix for window lengths 5 s and stride 10
![alt text](https://github.com/priack/Raia_exercise/blob/main/confusionMatrix_stride10.png)
The corresponding metris were: Accuracy: 0.705	 F1: 0.719	 Kappa: 0.726

As we can see, my initial observations were right. While the performance for 10 or 15 s is slightly better, this would introduce a 2.5 / 5 s delay on the prediction. Also, 15 s starts to decrease in the F1 score. Regarding the stride, we can see that using a lower one does not only do not lead to improvements, but it actually decreases the performance, while at the same time increases the training time.

Finally, I wanted to see what the performance was with the pre-defined features provided. This is the confusion matrix:
![alt text](https://github.com/priack/Raia_exercise/blob/main/confusionMatrix_predefined.png)
The corresponding metris were: Accuracy: 0.651	 F1: 0.684	 Kappa: 0.672

While there was no feature selection (much more important in this case due the the 42 features compared to the 8 in my previous analysis), in the previous analysis there was neither. Furthermore, the training time was higher, even if not in a really significant way (both cases in the order of seconds). We can see that the pre-defined features have a worse performance all across the board, including mixing motion activities (Upstairs) with inmobile ones (Sitting and Standing). The only case where the performance was slightly better was for the correct classification of Sitting and Downstairs. For every other activity the performance was worse. The rest of the metrics show a significant decrease as well.
