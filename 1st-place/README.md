![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/fog_mountain_from_site.jpg)
# From Fog Nets to Neural Nets - 1st Place
<br> <br>

# Entrant Background and Submission Overview

### Mini-bio
I started programming in the 1970s on my Dad’s HP calculator (up to 50 instructions) and a big terminal with ink and lined paper that my mother sometimes brought home in its suitcase from the university. I took my first programming course and spent long summer nights programming on the PLATO system, which offered graphics, touch-screens and music. After exploring other topics, I returned to computer science as a graduate student. Lately, I’ve mostly abandoned programming and turned to data analysis and statistics in the context of biometrics research.

### High Level Summary of Submission
I developed the models initially in JMP™ 12 because I’m familiar with that, didn’t have any great alternatives, and happened to learn about the contest a couple weeks after it started. Knowing I would have to provide my solution as open source, I limited myself to using ordinary least squares regression. I spent a lot of time visualizing the data to understand what was going on, get an intuitive sense for pattern vs. noise, and to select some thresholds. Soon I decided that winning would require dotting i's and crossing t’s, so I invested a lot of effort setting things up (e.g., imputing missing values). I tried various approaches and gradually learned; I submitted many entries. Eventually, after discovering strengths and weaknesses of the data and better understanding the gaps (missing data), I decided to make a single macro model by building a foundation of X variables that was much more complete than what was provided: more than just simple imputing, I forecast the variables I needed, especially through the night when some of the best variables were not reported. As a result, I needed only two models, macro and micro, rather than multiple macro models which was where I started (different models to match whichever combination of data sources was available for a prediction).

Early on I made a submission just to test the waters and decide whether to invest in the competition. I had only opened a couple of the raw data files, hadn’t read the official rules (they were difficult to find!), and with all the gaps in the data didn’t see an easy way to produce all of the predictions. That led me to develop a “diurnal” model, which simply predicts from time. That proved very useful. I kept trying to remove it from the final solution, but it insisted on staying. I tended to use that as a baseline, then multiply it up or down using the other predictors.

Another key strategy was recognizing dry conditions and keeping those events out of the regressions that I really wanted to swing way high during wet spells. Eventually I clued in on the near irrelevance of errors during dry spells; I hadn’t worked with RMSE before.

### Omitted Work
Building the regression models in layers worked much better than attempts at combining more factors at once. Specifically, I settled on a strategy of incorporating strong variables first, then adding on progressively weaker variables.

The variables that create the “splits” (THL and DrySplit) I also tried as factors in the regressions. It worked much better to split the task into multiple separate regressions based on these.

### Tools Used
JMP. It’s interactive visualization capabilities were important, especially the way the data is dynamically linked to the tables. For example, points selected in a scatterplot appear as rows selected in the tables.

I did not do any programming or scripting during the modeling phase: nothing more than column formulas.

### Model Evaluation
One thing I did not do was validation, such as holdout or cross-validation. I tried holdouts briefly early on and found the RMSE numbers all over the place, the procedure troublesome, and the results confusing. That lesson was invaluable, but learnt quickly.

I kept an eye on p-values when deciding whether to keep terms. Not sure what they mean with time-based samples, but if they weren’t strong I generally moved on.

I relied heavily on intuition and just playing in the data.

The best method I used was simply plotting the predictions against the actuals against time on the x-axis. I polished that a bit with some standard labeling, splitting the long timeline into sections, overlaying markers indicating where the training sets were, etc. When predictions were off, I tried to track down variables that could be used to improve the model; typically, I’d focus on a brief weather event and try to discover clues as to why I had not predicted it accurately.

The better the models got, the more I relied on submitting little incremental changes to make sure I stayed on course. My big clever ideas often combined gains and losses, and I couldn’t sort out which was which. I resisted the temptation to overfit the test by ratcheting random gains; hopefully that’s evident in the solution. Wind in the Micro model may be an exception (see below).

### Potentially Helpful Features
There’s something I can’t figure out in the Micro model related to winds. I think something anomalous occurred; the behavior and solution seem totally illogical. Another year of data would help resolve this. Maybe it just requires someone with some understanding of meteorology looking at the yield data for the test samples. I took advantage of the “split” model (separate regressions based on the nominal THL variable) to model the High (H) and Very High (VH) conditions differently to get around this (see code). I believe this part of the solution is poorly modeled and fit to the test.

The Micro model uses Sidi Ifni cloud data (Nh). This greatly displeased me, because it’s a separate data stream, but winning was paramount.

### Notes About the Model
A couple of the programs run slowly because I used for-loops, having given up on getting python to cooperate more efficiently (my ignorance of python).

My handling of missing data is not fully robust. It might fail to produce a prediction now and then.

There are some comments in the code. This was my first python program and I clearly didn’t know what I was doing in some cases, but I got it to work.

### Future Steps
RMSE is nice for a competition, but I doubt it really captures your ideal requirements. So I’d start with a conversation about that.

For the Micro model, I intended, but never got around to predicting yield from each 5-minute interval, then summing those to get the 2-hour prediction. Instead I used the standard deviation in 5-minute leaf wetness measurements over the past 2 hours. My solution worked well, but I expect integrating should work better.

I prefer logical, elegant, physical models, but especially in the Macro data, I’ll bet you could greatly improve RMSE simply by fusing the output of competing solutions (use my prediction and someone else’s as two inputs to predict yield). There’s a good chance you could get a big boost quickly. I had such a fusion in my back pocket for use at the last minute, but I’m glad I didn’t submit it (ugly and it would have complicated porting).

Another way to fuse would be to take my approach to preparing the data (forecasting) and combine that with someone else’s fancy neural nets, deep learning, or whatever.

I would put more grunt work into the forecasting. Garbage in, garbage out. For example, I didn’t forecast T-Td over night as well as I could have. Generally my forecast models are very simple, just enough to have something in place.

I didn’t understand knotted splines deeply as I do now after porting and running into several difficulties. JMP places the knots at equidistant intervals over the x-range; the user can only control the number of knots. In some cases I used up to 9 knots to get a tight response over a narrow range of x-values, e.g., the sweet spot for leaf wetness around 500-550, or StdDev(leaf wetness) very close to zero. These could be fit better with fewer, better placed knots. Similarly, the diurnal model is inherently cyclic (continuous at midnight): I couldn’t express that in JMP, but saw a way in patsy. I don’t know how much improvement is possible because I worked hard to get good fits, but I went about it all wrong.

In many cases, I used a very simple (sub)model for robustness, whereas a more complex model might have been more accurate. Some of the splines with 3 knots are examples. Another year of data would be the easiest basis for improving these.

That cloud data is rich and not fully exploited. I relied mostly on inferring from the data. A brief conversation with a meteorologist would be useful. For example, when there were very few learning instances of the many descriptors like rain, mist, showers and thunderstorms (in specific combinations with the other variables), I punted.

I would simplify and tweak the “splits,” i.e., the micro variable THL and the macro variable DrySplit. I got these to work OK, but it was too disruptive among many things I was trying to also play much with these. For starters, I’d combine THL “dry” and “M”; that split can’t be worth much and it complicates the model.

<br><br>
# Replicating the Submission

This code was developed on a Mac, OS X El Capitan.

### Install Python (version 2.7.10)
* Packages:
    * pandas
    * numpy
    * datetime
    * statsmodels.api
    * math
    * csv
    * re
    * subprocess


1. Put the 9 raw csv files in code/data/raw
    * Macrocliamte Sidi Ifni Weather Station.csv
    * Macroclimate Agadir Airport.csv
    * Macroclimate Guelmim Airport.csv
    * Target Variable Water Yield.csv
    * Test set Microclimate (2 hour intervals).csv
    * Test set Microclimate (5 min intervals).csv
    * Training set Microclimate (2 hour intervals).csv
    * Training set Microclimate (5 minute intervals).csv
    * VISUALIZATION Water Consumption Data.csv
    * submission_format.csv

2. The following python files will be in code/src/models:
    * `A_StructureInputFiles.py`
    * `B_Forecast.py`
    * `C_DeriveColumns.py`
    * `D_Regress.py`
    * `E_LWS_Step.py`
    * `F_StdDev_LWS.py`
    * `G_MicroRegress.py`
    * `H_CombineModels.py`

3. Run `fog.py`, which is located in `/code/`
    * This may take a minute to complete: steps B and F are slow (implemented with for-loops); E complains a lot ("copy of a slice"), but succeeds.

    The final output is written to `code/data/Submission.csv`
