![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/fog_mountain_from_site.jpg)
# From Fog Nets to Neural Nets - 2nd Place
<br> <br>

# Entrant Background and Submission Overview

### Mini-bio
I am generally interested in data analysis, and seeking original ways to understand complex systems,having first started doing so in computational biology. What drew me in particular to this competition was the fogwater project that I had previously heard of, and that I find really exciting.

### High Level Summary of Submission
I used a random forest algorithm, with some feature engineering. I used only microclimtate data when it was available and macroclimate data from guelmim when it was not, notably cloudiness, humidity, temperature, wind data and temporal variables like day of the year and hour of the day. For circular values, like day of the year or wind direction, I used the cosinus and sinus transforms. I imputed missing values in microclimate and macroclimate data by using the last seen value so as not to pollute with future data. I trained and predicted separately each subset of the test set, using for subset between times t1 and t2 the training set available for t<=t1.

### Omitted Work
I tried various combinations of predicting variables. I have left the ones I didn't end up using in the code, so that they can still be tried. I tried neural nets with little luck.

### Tools Used
I did various exploratory data analyses, plotting variables against time, and doing scatterplots (using function smoothScatter in R).

### Model Evaluation
Given the particular nature of the test set and advice not to use future data for training, there was a discrepancy between the usefulness of a model when the training set was large versus when it was small. During cross-validation, I relied more on the error estimates of the random forest model than on the rmse I calculated.

### Potentially Helpful Features
Microclimate measures of humidity and leafwet sensors are extremely correlated with the yield. It could be useful to have sensing material with a higher definition for high values, if that is possible.

### Future Steps
If a large training set could be used for prediction, I would probably try to further optimize the choice of predicting variables. Also, the random forest algorithm behaved quite well with a set of highly correlated variables, but I would look further using other similar methods to see if they behaved better.

<br><br>
# Replicating the Submission

### I. Install R
* https://www.r-project.org/

### II. Open R:

1. Set working environement to project root: setwd("<appropriate
    path>");

2. Install necessary packages: install.packages("randomForest");
    install.packages("getopt");

3. Charge user-defined functions: source("src/functions.R");

4. Read raw data, transform it and save it in directory data_made/ :
    source("src/make_dataset.R");

5. Read files from directory data_made/, perform training and
    prediction (description of the training process will be printed in
    std output while the prediction result will be printed in output
    file (here output/prediction.csv)):
    system("Rscript src/train_and_predict.R -o output/prediction.csv");
