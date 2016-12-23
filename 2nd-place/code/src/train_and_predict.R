library(randomForest);
library(getopt);

## Read argument output file for prediction result

args=matrix(c("output","o",1,"character"),byrow=T,ncol=4);
opt=getopt(args);
outfile=opt$output;

## Read computed datasets resulting from the execution of make_dataset.R, and convert time columns to POSIXct objects
train2hga=read.csv("data_made/train2h_guelmim_agadir.csv");
test2hga=read.csv("data_made/test2h_guelmim_agadir.csv");

train2hga$X=as.POSIXct(train2hga$X,format="%Y-%m-%d %H:%M:%S",tz="WET");
test2hga$X=as.POSIXct(test2hga$X,format="%Y-%m-%d %H:%M:%S",tz="WET");

## Read submission format file file, and convert time column to POSIXct object

testing=read.csv("data/submission_format.csv");
testing$X=as.POSIXct(testing$X,format="%Y-%m-%d %H:%M:%S",tz="WET");


## In the test set, identify the different windows that are not consecutive in time. Keep the index boundaries in object "index", and the starting times of each window in object "starts"

nt=dim(test2hga)[1];
index=0;
starts=test2hga$X[1];
for(i in 1:(nt-1))
{
    if(test2hga$X[i+1]-test2hga$X[i]>3)
    {
         index=c(index,i);
         starts=c(starts,test2hga$X[i+1]);
    }

}
index=c(index,nt);
count=length(index);


## For each window in the test set, perform training and then prediction.

predtot=0;
for(i in 1:(count-1))
{

    ## If microclimate data is available, the model (rf) uses only microclimate data.

    if(test2hga$na[index[i]+1]==0) #microData available
    {
        test=test2hga[(index[i]+1):index[i+1],];
        train=na.omit(train2hga[train2hga$X<=starts[i],]);
        cat("WINDOW",i,": Dimension of training set is",dim(train)[1],"\n","Dim of testing set is",dim(test)[1],"\n");
        cat("Test start",index[i]+1,"Test end",index[i+1],"\n");
        cat("Startdate",as.character(starts[i]),"\n");
        rf=randomForest(yield~humidity+temp+leafwet450_min+leafwet_lwscnt+gusts_ms+hourcos+hoursin+ydaycos+ydaysin+winddircos+winddirsin,mtry=5,data=train);
        print(rf);
        cat("\n");
        pred=predict(rf,test);
    }

    ## If microclimate data is not available, the model (rf2) uses macroclimate data from guelmim.

    else if(test2hga$na[index[i]+1]==1)
    {
        test=test2hga[(index[i]+1):index[i+1],];
        trainmacro=na.omit(train2hga[train2hga$X<=starts[i],]);
        cat("WINDOW",i,": Dimension of trainmacro dataset is",dim(trainmacro)[1],"\n","Dim of testing set is",dim(test)[1],"\n");
        cat("Test start",index[i]+1,"Test end",index[i+1],"\n");
        cat("Startdate",as.character(starts[i]),"\n");

        ## Particular case for the first window where the size of the training set is null, then there is no training, and predicted values are set to value 0.
        if(dim(trainmacro)[1]==0)
        {
            pred=rep(0,dim(test)[1]);
            cat("No trainmacro data\n");
        }

        else
        {
            rf2=randomForest(yield~U.g+Uavg5.g+cloud480.g+cloud.g+Udiff5.g+Ff.g+windsin.g+windcos.g+P.g+hourcos+hoursin+ydaycos+ydaysin,mtry=4,data=trainmacro);
            print(rf2);
            cat("\n");
            pred=predict(rf2,test);


        }
    }


    predtot=c(predtot,pred);
}
predtot=predtot[-1];

## Replace negative values of prediction by 0.
predtot[sign(predtot)==-1]=0;

## Construct a data frame with time and prediction
predres=data.frame(testing$X,predtot);
names(predres)=c("","yield");

## Write prediction datav frame into a file
write.csv(predres,file=outfile,row.names=F,quote=F);
