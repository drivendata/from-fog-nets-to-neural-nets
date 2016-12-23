
## Function rowavg receives as parameters a vector x and an integer n, computes for each row of x the average of the row and the previous n rows, and returns a vector x2 of the same length as x containing the computed values

rowavg = function(x,n)
{
    l=length(x);
    x2=0;

    for(i in 1:n)
    {
        avg=mean(x[1:i],na.rm=T);
        x2=c(x2,avg);
    }
    for(i in (n+1):l)
    {
        avg=mean(x[(i-n):i],na.rm=T);
        x2=c(x2,avg);
    }
    x2=x2[-1];

    return(x2);
}

## Function lastValue receives as a parameter a vector x, and returns a vector resv of the same length as x, where NA values have been replaced by the closest previous non-NA value in the vector.

lastValue = function(x)
{
    index=!is.na(x);
    vals=c(NA,x[index]);
    fillindex=cumsum(index)+1;
    resv=vals[fillindex];
    return(resv)
}

## Function rmse receives as parameters two vectors x and y, and returns the root-mean-square error
rmse = function(x,y)
{
    resv=sqrt(sum((x-y)^2)/length(x))
    return(resv);

}
