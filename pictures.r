
## Get package implementing Weinstein et al., from GitHub
library(devtools)
install_github("selectiveCI", "johnros", subdir='selectiveCI')
library(selectiveCI)

library(raster)


x <- scale(matrix(c(1,1,1.1,.9,.8,1.2),2),center=FALSE)


b1 <- (x[2,1]+x[2,2])/(x[1,1]+x[1,2])
b2 <- (x[2,1]+x[2,3])/(x[1,1]+x[1,3])

ci.len.wrapper <- function(xy) {
    good <- which((xy[,2] > b1*xy[,1]) & (xy[,2] < b2 * xy[,1]))
    z <- rep(NA,nrow(xy))
    z[good] <- ci.len(xy[good,1],xy[good,2])
    z
}

ci.len <- function(x,y) {
    cutoff.x <- ifelse(x>y,(y-x)/(b1-1),(y-x)/(b2-1))
    cutoff.y <- ifelse(x>y,b1*(y-x)/(b1-1),b2*(y-x)/(b2-1))
    cutoff <- (cutoff.x + cutoff.y)/sqrt(2)
    observed <- (x+y)/sqrt(2)
    apply(cbind(observed,cutoff),1,function(x) {
        ci <- try(ShortestCI(x[1],1,x[2],.05),silent=TRUE)
        if(is.list(ci)) {
            return(ci$upper - ci$lower)
        } else {
            return(NA)
        }
    })
}

xy <- expand.grid(c(-1,seq(0,4,.02)),c(-1,seq(0,4,.02)))
z <- ci.len.wrapper(xy)
## This is a hack because of a bug in the package
z[xy[,1] > 0 & abs(xy[,1]-xy[,2])<.023] <- 2*1.96


rast <- rasterFromXYZ(cbind(xy,z))

pdf("CILengthCorr.pdf")
plot(rast,xlim=c(-2.5,4),ylim=c(-2.5,4),xlab=expression(y[1]),ylab=expression(y[2]),col=rev(heat.colors(20)),
     main="CI Length for Univariate Model")
abline(h=0,lty=3,col="gray")
abline(v=0,lty=3,col="gray")
arrows(x0=c(0,0,0),y0=c(0,0,0),x1=c(x[1,]),y1=c(x[2,]),length=.15)
abline(0,b1,lty=2)
abline(0,b2,lty=2)
dev.off()




