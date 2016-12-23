## Read the guelmim and agadir macrocimate files and create time column which is a POSIXct object

agadir=read.csv("data/Macroclimate-Agadir.csv");
agadir$X=as.POSIXct(agadir$Local.time.in.Agadir...Al.Massira..airport.,format="%Y-%m-%d %H:%M:%S",tz="WET");
#
guelmim=read.csv("data/Macroclimate-Guelmim.csv");
guelmim$X=as.POSIXct(guelmim$Local.time.in.Guelmim,format="%Y-%m-%d %H:%M:%S",tz="WET");

## Remove original time column

agadir=agadir[,-1];
guelmim=guelmim[,-1];

## Convert wind direction in guelmim from a label to a value between 0 and 360, with the convention that 0 corresponds to North

guelmim$winddir=rep(0,dim(guelmim)[1]);
for(i in 1:length(guelmim$winddir))
{
    if (guelmim$DD[i]=="Calm, no wind") {guelmim$winddir[i]=NA}
    else if (guelmim$DD[i]=="variable wind direction") {guelmim$winddir[i]=NA}
    else if (guelmim$DD[i]=="Wind blowing from the east") {guelmim$winddir[i]=90}
    else if (guelmim$DD[i]=="Wind blowing from the west") {guelmim$winddir[i]=270}
    else if (guelmim$DD[i]=="Wind blowing from the north") {guelmim$winddir[i]=0}
    else if (guelmim$DD[i]=="Wind blowing from the south") {guelmim$winddir[i]=180}

    else if (guelmim$DD[i]=="Wind blowing from the north-east") {guelmim$winddir[i]=45}
    else if (guelmim$DD[i]=="Wind blowing from the north-west") {guelmim$winddir[i]=315}
    else if (guelmim$DD[i]=="Wind blowing from the south-east") {guelmim$winddir[i]=135}
    else if (guelmim$DD[i]=="Wind blowing from the south-west") {guelmim$winddir[i]=225}

    else if (guelmim$DD[i]=="Wind blowing from the north-northeast") {guelmim$winddir[i]=22.5}
    else if (guelmim$DD[i]=="Wind blowing from the north-northwest") {guelmim$winddir[i]=337.5}
    else if (guelmim$DD[i]=="Wind blowing from the south-southeast") {guelmim$winddir[i]=157.5}
    else if (guelmim$DD[i]=="Wind blowing from the south-southwest") {guelmim$winddir[i]=202.5}

    else if (guelmim$DD[i]=="Wind blowing from the east-northeast") {guelmim$winddir[i]=67.5}
    else if (guelmim$DD[i]=="Wind blowing from the west-northwest") {guelmim$winddir[i]=292.5}
    else if (guelmim$DD[i]=="Wind blowing from the east-southeast") {guelmim$winddir[i]=112.5}
    else if (guelmim$DD[i]=="Wind blowing from the west-southwest") {guelmim$winddir[i]=247.5}

}

## Convert wind direction in agadir from a label to a value between 0 and 360, with the convention that 0 corresponds to North

agadir$winddir=rep(0,dim(agadir)[1]);
for(i in 1:length(agadir$winddir))
{
    if (agadir$DD[i]=="Calm, no wind") {agadir$winddir[i]=NA}
    else if (agadir$DD[i]=="variable wind direction") {agadir$winddir[i]=NA}
    else if (agadir$DD[i]=="Wind blowing from the east") {agadir$winddir[i]=90}
    else if (agadir$DD[i]=="Wind blowing from the west") {agadir$winddir[i]=270}
    else if (agadir$DD[i]=="Wind blowing from the north") {agadir$winddir[i]=0}
    else if (agadir$DD[i]=="Wind blowing from the south") {agadir$winddir[i]=180}

    else if (agadir$DD[i]=="Wind blowing from the north-east") {agadir$winddir[i]=45}
    else if (agadir$DD[i]=="Wind blowing from the north-west") {agadir$winddir[i]=315}
    else if (agadir$DD[i]=="Wind blowing from the south-east") {agadir$winddir[i]=135}
    else if (agadir$DD[i]=="Wind blowing from the south-west") {agadir$winddir[i]=225}

    else if (agadir$DD[i]=="Wind blowing from the north-northeast") {agadir$winddir[i]=22.5}
    else if (agadir$DD[i]=="Wind blowing from the north-northwest") {agadir$winddir[i]=337.5}
    else if (agadir$DD[i]=="Wind blowing from the south-southeast") {agadir$winddir[i]=157.5}
    else if (agadir$DD[i]=="Wind blowing from the south-southwest") {agadir$winddir[i]=202.5}

    else if (agadir$DD[i]=="Wind blowing from the east-northeast") {agadir$winddir[i]=67.5}
    else if (agadir$DD[i]=="Wind blowing from the west-northwest") {agadir$winddir[i]=292.5}
    else if (agadir$DD[i]=="Wind blowing from the east-southeast") {agadir$winddir[i]=112.5}
    else if (agadir$DD[i]=="Wind blowing from the west-southwest") {agadir$winddir[i]=247.5}

}




## In guelmim data, from the variable c which corresponds to cloudiness, extract 5 variables:
## cloud: corresponds to maximum cloud cover regardless of altitude, with a value of 1 where cloud cover is 100%, a value of 0.75 where cloud copver is 60-90%, a value of 0.45 where cloud cover is 40-50%, a value of 0.2 where cloud cover is 10-30%, a value of NA where cloud cover is NA, and a value of 0 elsewhere.
## cloud180: corresponds to maximum cloud cover at an altitude of 180m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud300: corresponds to maximum cloud cover at an altitude of 300m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud480: corresponds to maximum cloud cover at an altitude of 480m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud6000: corresponds to maximum cloud cover at an altitude of 6000m, with possible values of 0,0.2,0.45,0.75,1 and NA

guelmim100=1*grepl("100%",guelmim$c,fixed=T);
guelmim75=0.75*grepl("60-90%",guelmim$c,fixed=T);
guelmim45=0.45*grepl("40-50%",guelmim$c,fixed=T);
guelmim20=0.2*grepl("10-30%",guelmim$c,fixed=T);
guelmimall=cbind(guelmim100,guelmim75,guelmim45,guelmim20);
guelmim$cloud=apply(guelmimall,1,max);
guelmim$cloud[guelmim$c=='']=NA;

guelmim100180=1*grepl("100%) 180",guelmim$c,fixed=T);
guelmim75180=0.75*grepl("60-90%) 180",guelmim$c,fixed=T);
guelmim45180=0.45*grepl("40-50%) 180",guelmim$c,fixed=T);
guelmim20180=0.2*grepl("10-30%) 180",guelmim$c,fixed=T);
guelmimall180=cbind(guelmim100180,guelmim75180,guelmim45180,guelmim20180);
guelmim$cloud180=apply(guelmimall180,1,max);
guelmim$cloud180[guelmim$c=='']=NA;

guelmim100300=1*grepl("100%) 300",guelmim$c,fixed=T);
guelmim75300=0.75*grepl("60-90%) 300",guelmim$c,fixed=T);
guelmim45300=0.45*grepl("40-50%) 300",guelmim$c,fixed=T);
guelmim20300=0.2*grepl("10-30%) 300",guelmim$c,fixed=T);
guelmimall300=cbind(guelmim100300,guelmim75300,guelmim45300,guelmim20300);
guelmim$cloud300=apply(guelmimall300,1,max);
guelmim$cloud300[guelmim$c=='']=NA;

guelmim100480=1*grepl("100%) 480",guelmim$c,fixed=T);
guelmim75480=0.75*grepl("60-90%) 480",guelmim$c,fixed=T);
guelmim45480=0.45*grepl("40-50%) 480",guelmim$c,fixed=T);
guelmim20480=0.2*grepl("10-30%) 480",guelmim$c,fixed=T);
guelmimall480=cbind(guelmim100480,guelmim75480,guelmim45480,guelmim20480);
guelmim$cloud480=apply(guelmimall480,1,max);
guelmim$cloud480[guelmim$c=='']=NA;

guelmim1006000=1*grepl("100%) 6000",guelmim$c,fixed=T);
guelmim756000=0.75*grepl("60-90%) 6000",guelmim$c,fixed=T);
guelmim456000=0.45*grepl("40-50%) 6000",guelmim$c,fixed=T);
guelmim206000=0.2*grepl("10-30%) 6000",guelmim$c,fixed=T);
guelmimall6000=cbind(guelmim1006000,guelmim756000,guelmim456000,guelmim206000);
guelmim$cloud6000=apply(guelmimall6000,1,max);
guelmim$cloud6000[guelmim$c=='']=NA;

## In guelmim data, convert VV variable to a numeric variable called vis.

guelmim$vis=as.numeric(gsub("[^\\d.]+", "", guelmim$VV, perl=TRUE));

## In guelmim data, remove unwanted columns

guelmim=subset(guelmim,select=-c(c,DD,ff10,Td,WW,W.W.,P0,VV));


## In agadir data, from the variable c which corresponds to cloudiness, extract 5 variables:
## cloud: corresponds to maximum cloud cover regardless of altitude, with a value of 1 where cloud cover is 100%, a value of 0.75 where cloud copver is 60-90%, a value of 0.45 where cloud cover is 40-50%, a value of 0.2 where cloud cover is 10-30%, a value of NA where cloud cover is NA, and a value of 0 elsewhere.
## cloud300: corresponds to maximum cloud cover at an altitude of 300m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud480: corresponds to maximum cloud cover at an altitude of 480m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud600: corresponds to maximum cloud cover at an altitude of 600m, with possible values of 0,0.2,0.45,0.75,1 and NA
## cloud3000: corresponds to maximum cloud cover at an altitude of 3000m, with possible values of 0,0.2,0.45,0.75,1 and NA

agadir100=1*grepl("100%",agadir$c,fixed=T);
agadir75=0.75*grepl("60-90%",agadir$c,fixed=T);
agadir45=0.45*grepl("40-50%",agadir$c,fixed=T);
agadir20=0.2*grepl("10-30%",agadir$c,fixed=T);
agadirall=cbind(agadir100,agadir75,agadir45,agadir20);
agadir$cloud=apply(agadirall,1,max);
agadir$cloud[agadir$c=='']=NA;

agadir100300=1*grepl("100%) 300",agadir$c,fixed=T);
agadir75300=0.75*grepl("60-90%) 300",agadir$c,fixed=T);
agadir45300=0.45*grepl("40-50%) 300",agadir$c,fixed=T);
agadir20300=0.2*grepl("10-30%) 300",agadir$c,fixed=T);
agadirall300=cbind(agadir100300,agadir75300,agadir45300,agadir20300);
agadir$cloud300=apply(agadirall300,1,max);
agadir$cloud300[agadir$c=='']=NA;

agadir100480=1*grepl("100%) 480",agadir$c,fixed=T);
agadir75480=0.75*grepl("60-90%) 480",agadir$c,fixed=T);
agadir45480=0.45*grepl("40-50%) 480",agadir$c,fixed=T);
agadir20480=0.2*grepl("10-30%) 480",agadir$c,fixed=T);
agadirall480=cbind(agadir100480,agadir75480,agadir45480,agadir20480);
agadir$cloud480=apply(agadirall480,1,max);
agadir$cloud480[agadir$c=='']=NA;

agadir100600=1*grepl("100%) 600",agadir$c,fixed=T);
agadir75600=0.75*grepl("60-90%) 600",agadir$c,fixed=T);
agadir45600=0.45*grepl("40-50%) 600",agadir$c,fixed=T);
agadir20600=0.2*grepl("10-30%) 600",agadir$c,fixed=T);
agadirall600=cbind(agadir100600,agadir75600,agadir45600,agadir20600);
agadir$cloud600=apply(agadirall600,1,max);
agadir$cloud600[agadir$c=='']=NA;

agadir1003000=1*grepl("100%) 3000",agadir$c,fixed=T);
agadir753000=0.75*grepl("60-90%) 3000",agadir$c,fixed=T);
agadir453000=0.45*grepl("40-50%) 3000",agadir$c,fixed=T);
agadir203000=0.2*grepl("10-30%) 3000",agadir$c,fixed=T);
agadirall3000=cbind(agadir1003000,agadir753000,agadir453000,agadir203000);
agadir$cloud3000=apply(agadirall3000,1,max);
agadir$cloud3000[agadir$c=='']=NA;

## In agadir data, convert VV variable to a numeric variable called vis.

agadir$vis=as.numeric(gsub("[^\\d.]+", "", agadir$VV, perl=TRUE));

## In agadir data, remove unwanted columns

agadir=subset(agadir,select=-c(c,DD,ff10,WW,W.W.,Td,P0,VV));



## Read submission format file and training set 2h file, and convert time column to POSIXct object

testing=read.csv("data/submission_format.csv");
testing$X=as.POSIXct(testing$X,format="%Y-%m-%d %H:%M:%S",tz="WET");

training=read.csv("data/Trainingset-microclimate-2hinterval.csv");
training$X=as.POSIXct(training$X,format="%Y-%m-%d %H:%M:%S",tz="WET");

## Combine times in training and testing set, and perform a left join with guelmim data, so that the newly created guelmim data has a time column containing all training and testing times

alltimes=data.frame(X=sort(c(testing$X,training$X)));
alltimes$X=as.POSIXct(alltimes$X,format="%Y-%m-%d %H:%M:%S",tz="WET");

guelmim=merge(alltimes,guelmim,by="X",all=T);

## Replace NA values in guelmim data with the last seen non-NA value using the lastValue function (see functions.R)

guelmim$U=lastValue(guelmim$U);
guelmim$Ff=lastValue(guelmim$Ff);
guelmim$T=lastValue(guelmim$T);
guelmim$P=lastValue(guelmim$P);
guelmim$winddir=lastValue(guelmim$winddir);
guelmim$cloud=lastValue(guelmim$cloud);
guelmim$cloud180=lastValue(guelmim$cloud180);
guelmim$cloud300=lastValue(guelmim$cloud300);
guelmim$cloud480=lastValue(guelmim$cloud480);
guelmim$cloud6000=lastValue(guelmim$cloud6000);
guelmim$vis=lastValue(guelmim$vis);

## Calculate variable Uavg5, which corresponds for each row to the average of U (humidity) with the previous 4 values of U, using function rowavg (see functions.R).
## Calculate variable Uavg10, which corresponds for each row to the average of U (humidity) with the previous 9 values of U, using function rowavg (see functions.R).
## Calculate variavle Udiff5, which is the difference between U and Uavg5

guelmim$Uavg5=rowavg(guelmim$U,4);
guelmim$Uavg10=rowavg(guelmim$U,9);
guelmim$Udiff5=guelmim$U-guelmim$Uavg5;

## Calculate variables windcos and windsin, which correspond to the cosinus and sinus of the wind direction

guelmim$windcos=cos(guelmim$winddir*pi/180);
guelmim$windsin=sin(guelmim$winddir*pi/180);


## Perform a left join of all times with agadir data, so that the newly created agadir data has a time column containing all training and testing times

agadir=merge(alltimes,agadir,by="X",all=T);

## Replace NA values in agadir data with the last seen non-NA value using the lastValue function (see functions.R)

agadir$U=lastValue(agadir$U);
agadir$Ff=lastValue(agadir$Ff);
agadir$T=lastValue(agadir$T);
agadir$P=lastValue(agadir$P);
agadir$winddir=lastValue(agadir$winddir);
agadir$cloud=lastValue(agadir$cloud);
agadir$cloud300=lastValue(agadir$cloud300);
agadir$cloud480=lastValue(agadir$cloud480);
agadir$cloud600=lastValue(agadir$cloud600);
agadir$cloud3000=lastValue(agadir$cloud3000);
agadir$vis=lastValue(agadir$vis);

## Calculate variable Uavg5, which corresponds for each row to the average of U (humidity) with the previous 4 values of U, using function rowavg (see functions.R).
## Calculate variable Uavg10, which corresponds for each row to the average of U (humidity) with the previous 9 values of U, using function rowavg (see functions.R).
## Calculate variavle Udiff5, which is the difference between U and Uavg5

agadir$Uavg5=rowavg(agadir$U,4);
agadir$Uavg10=rowavg(agadir$U,9);
agadir$Udiff5=agadir$U-agadir$Uavg5;

## Calculate variables windcos and windsin, which correspond to the cosinus and sinus of the wind direction

agadir$windcos=cos(agadir$winddir*pi/180);
agadir$windsin=sin(agadir$winddir*pi/180);

guelmim=as.data.frame(guelmim);
agadir=as.data.frame(agadir);

## Read microclimate 2h training set and compute hour, cosinus and sinus of the hour, day of the year, cosinus and sinus of the day of the year, cosinus and sinus of wind direction.

train2h=read.csv("data/Trainingset-microclimate-2hinterval.csv");
train2h$X=as.POSIXlt(as.character(train2h$X),format="%Y-%m-%d %H:%M:%S",tz="WET");

train2h$hourcos=cos(train2h$X$hour*pi/12);
train2h$hoursin=sin(train2h$X$hour*pi/12);
train2h$ydaycos=cos((train2h$X$yday+1)*2*pi/365);
train2h$ydaysin=sin((train2h$X$yday+1)*2*pi/365);
train2h$winddircos=cos(train2h$wind_dir*pi/180);
train2h$winddirsin=sin(train2h$wind_dir*pi/180);
train2h$hour=train2h$X$hour;


train2h$yday=train2h$X$yday+1;
train2h$X=as.POSIXct(train2h$X);

## Read file with water yield, and combine with training set

water=read.csv("data/TargetVariable-WaterYield.csv");
train2h$yield=water$yield;

## Remove unwanted columns from training set

train2h=subset(train2h,select=-c(leafwet460_min,percip_mm));

train2h=as.data.frame(train2h);

## Replace NA values in training data with the last seen non-NA value using the lastValue function (see functions.R)

for(i in 2:dim(train2h)[2])
{
    train2h[,i]=lastValue(train2h[,i]);
}

## Perform a left join between training data and respectively guelmim data and agadir data

train2hguelmim=merge(train2h,guelmim,by="X",all.x=T);
train2hagadir=merge(train2h,agadir,by="X",all.x=T);

## Perform a left join between on one side the joining of training data and guelmim data and on the other side agadir data

train2hga=merge(train2hguelmim,agadir,by="X",all.x=T,suffixes=c(".g",".a"));



## Read microclimate test file
microtesting=read.csv("data/Testset-microclimate-2hinterval.csv");
microtesting$X=as.POSIXct(microtesting$X,format="%Y-%m-%d %H:%M:%S",tz="WET");

## Perform a left join between the times in the submission format and in the microclimate test data

test2h=merge(testing,microtesting,by="X",all.x=T);

## Remove unwanted variables from test data

test2h=subset(test2h,select=-c(yield,leafwet460_min,percip_mm));

## Compute hour, cosinus and sinus of the hour, day of the year, cosinus and sinus of the day of the year, cosinus and sinus of wind direction

test2h$winddircos=cos(test2h$wind_dir*pi/180);
test2h$winddirsin=sin(test2h$wind_dir*pi/180);

test2h$X=as.POSIXlt(test2h$X);

test2h$hour=test2h$X$hour;
test2h$hourcos=cos(test2h$X$hour*pi/12);
test2h$hoursin=sin(test2h$X$hour*pi/12);
test2h$yday=test2h$X$yday+1;
test2h$ydaycos=cos(test2h$yday*2*pi/365);
test2h$ydaysin=sin(test2h$yday*2*pi/365);

## Add to test set a column na, where a value of 0 means there is no microclimate data available and a value of 1 that there is.

test2h$na=as.numeric(is.na(test2h$temp));

test2h$X=as.POSIXct(test2h$X);

test2h=as.data.frame(test2h);

## Perform a left join between test data and respectively guelmim data and agadir data

test2hguelmim=merge(test2h,guelmim,by="X",all.x=T);
test2hagadir=merge(test2h,agadir,by="X",all.x=T);

## Perform a left join between on one side the joining of test data and guelmim data and on the other side agadir data

test2hga=merge(test2hguelmim,agadir,by="X",all.x=T,suffixes=c(".g",".a"));


## Write "train2hga" (training+agadir+guelmim) variable in file "train2h_guelmim_agadir.csv"
write.csv(train2hga,file="data_made/train2h_guelmim_agadir.csv",quote=F,row.names=F);


## Write "test2hga" (test+agadir+guelmim) variable in file "test2h_guelmim_agadir.csv"
write.csv(test2hga,file="data_made/test2h_guelmim_agadir.csv",quote=F,row.names=F);
