[<img src='https://drivendata.s3.amazonaws.com/images/drivendata.png'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/fog_mountain_from_site.jpg)

# From Fog Nets to Neural Nets
## Goal of the Competition
Dar Si Hmad (DSH) has built the world’s largest fog-collection and distribution system to serve these indigenous communities. Delivery of fogwater significantly reduces women’s laborious water-gathering chores, and helps foster stable communities, continuation of ancestral languages and ways of living in thriving local environments.

Your challenge is to develop a model that will predict the yield of DSH’s fog nets for every day during an evaluation period, using historical data about meteorological conditions and the fog net installations. Accurate predictions will enable DSH to operate more effectively and the communities it serves to be have greater access to fresh water throughout the year.

## What's in this Repository
This repository contains code volunteered from leading competitors in the [From Fog Nets to Neural Nets](https://www.drivendata.org/competitions/9/) on DrivenData.

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | --- | ---
1 | ulery | 3.0239 | 2.9415 | I limited myself to using ordinary least squares regression. I spent a lot of time visualizing the data to understand what was going on, get an intuitive sense for pattern vs. noise, and to select some thresholds.
2 | lz01 | 3.0436 | 2.9694 | I used a random forest algorithm, with some feature engineering. I used only microclimtate data when it was available and macroclimate data from guelmim when it was not.
3 | oliviers | 3.0514 | 2.9723 | I made a program which allows to easily build some "mini models" which use only a part of the data and different aggregation techniques, and then choose the best performing "mini model" for each line of the submission.


#### Winner's Interview: ["Visualizing a Future for Fog Nets"](http://blog.drivendata.org/2016/07/19/fogwater-winner/)
