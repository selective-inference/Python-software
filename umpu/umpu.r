### Construct UMAU intervals for a truncated Gaussian




## Simple functions for specifying and checking a "cutoffs" object, which
##   is just a matrix, each of whose rows is a non-empty real interval (infinite endpoints allowed)

check.cutoffs <- function(cutoffs) {
    if(!is.matrix(cutoffs) || dim(cutoffs)[2] != 2) stop("cutoffs should be a matrix with 2 columns")
    if(sum(cutoffs[,2] <= cutoffs[,1]) > 0) stop("all right endpoints should be > left endpoints")
    if(sum(diff(c(t(cutoffs))) <= 0) > 0) stop("endpoints should be strictly increasing")
}

negate.cutoffs <- function(cutoffs) {
    -cutoffs[nrow(cutoffs):1,2:1]
}

negate.cutoffs(rbind(c(-Inf,-4),c(-3,-2),c(7,Inf)))

two.sided.cutoff <- function(x) rbind(neg=c(-Inf,-abs(x)),pos=c(abs(x),Inf))

two.sided.cutoff(3)

## Compute Phi(b-mu) - Phi(a-mu) in a numerically robust way
## mu can be a vector
pnorm.interval <- function(mu, ab) {
    ifelse(mean(ab) - mu < 0,
           pnorm(ab[2] - mu) - pnorm(ab[1] - mu),
           pnorm(mu - ab[1]) - pnorm(mu - ab[2]))
}

## Compute Phi(b-mu) - Phi(a-mu) for each [a,b] in S
pnorm.cutoffs <- function(mu, cutoffs) {
    ret <- apply(cutoffs, 1, function(cut) pnorm.interval(mu, cut))
    if(!is.matrix(ret)) ret <- t(ret)
    dimnames(ret) <- list(as.character(mu),row.names(cutoffs))
    ret
}

pnorm.cutoffs(-1:1,two.sided.cutoff(3))
pnorm.cutoffs(-1,two.sided.cutoff(3))


## Compute phi(b-mu) - phi(a-mu) for each [a,b] in S
## mu can be a vector
dnorm.cutoffs <- function(mu, cutoffs) {
    ret <- apply(cutoffs, 1, function(cut) dnorm(cut[2] - mu) - dnorm(cut[1] - mu))
    if(!is.matrix(ret)) ret <- t(ret)
    dimnames(ret) <- list(as.character(mu),row.names(cutoffs))
    ret
}

dnorm.cutoffs(-1:1,two.sided.cutoff(3))
dnorm.cutoffs(-1,two.sided.cutoff(3))

## Compute P_mu(X<x)
## mu CANNOT be a vector, pk is a one-row matrix or vector
## x must be in one of the intervals
F.mu <- function(x, mu, cutoffs, pk=pnorm.cutoffs(mu, cutoffs)) {
    stopifnot(length(mu)==1, nrow(pk)==1 || !is.matrix(pk))
    K <- length(pk)
    p <- sum(pk)
    k <- which(x >= cutoffs[,1] & x <= cutoffs[,2])
    stopifnot(length(k)==1)
    (sum(pk[(1:K) < k]) + pnorm.interval(mu, c(cutoffs[k,1], x)) ) / p
}

F.mu(-10.01,0,two.sided.cutoff(10))
F.mu(-9,0,two.sided.cutoff(10))
F.mu(10.2,0,two.sided.cutoff(10))

## Compute the inverse of the previous function
## mu CANNOT be a vector, pk is a one-row matrix or vector
F.inv.mu <- function(F, mu, cutoffs, pk=pnorm.cutoffs(mu, cutoffs)) {
    stopifnot(length(mu)==1, nrow(pk)==1 || !is.matrix(pk))
    p <- sum(pk)
    k <- max(which(c(0,cumsum(pk))/p < F))
    pnorm.increment <- p*F - c(0,cumsum(pk))[k]
    if(mean(cutoffs[k,]) < 0)
        mu + qnorm(pnorm(cutoffs[k,1]-mu) + pnorm.increment)
    else
        mu + qnorm(pnorm(cutoffs[k,1]-mu,lower.tail=FALSE) - pnorm.increment,lower.tail=FALSE)
}

## Compute c2(c1) for a single c1
## mu CANNOT be a vector, pk is a one-row matrix or vector
c2.single <- function(c1, mu, alpha, cutoffs, pk=pnorm.cutoffs(mu, cutoffs)) {
    stopifnot(length(mu)==1, nrow(pk)==1 || !is.matrix(pk))
    K <- length(pk)
    alpha1 <- F.mu(c1, mu, cutoffs, pk)
    if(alpha1 > alpha) return(NA)

    alpha2 <- alpha-alpha1
    return(F.inv.mu(1-alpha2, mu, cutoffs, pk))
}

c2.single(-10.3, 0, .05, two.sided.cutoff(10))
F.mu(-10.3, 0, two.sided.cutoff(10))

## Do the same, for a vector of c1 and mu
c2 <- function(c1, mu, alpha, cutoffs, pk=pnorm.cutoffs(mu, cutoffs)) {
    sapply(1:length(c1),function(i)
           c2.single(c1[i], mu[i], alpha, cutoffs,
                     pk[i,,drop=FALSE]))
}
c2(-10.3, 0, .05, two.sided.cutoff(10))


## Compute g_mu(c1) for a single mu and c1 (see LaTeX documentation)
## c1 and mu CANNOT be vectors, pk is NOT a matrix
g.mu.single <- function(c1, mu, alpha, cutoffs,
                        pk=pnorm.cutoffs(mu, cutoffs), dk=dnorm.cutoffs(mu, cutoffs)) {
    const <- (1-alpha) * (sum(-dk) + mu * sum(pk))
    cc2 <- c2(c1, mu, alpha, cutoffs, pk)
    if(is.na(cc2)) return(Inf)

    K <- length(pk)
    p <- sum(pk)
    k1 <- which(c1 >= cutoffs[,1] & c1 <= cutoffs[,2])
    stopifnot(length(k1)==1)
    k2 <- which(cc2 >= cutoffs[,1] & cc2 <= cutoffs[,2])
    stopifnot(length(k2)==1)

    if(k1 < k2) {
        sum(-dk[(1:K) > k1 & (1:K) < k2]) + mu * sum(pk[(1:K) > k1 & (1:K) < k2]) +
            - dnorm(cutoffs[k1,2] - mu) + dnorm(c1 - mu) - dnorm(cc2 - mu) + dnorm(cutoffs[k2,1] - mu) +
                mu * (pnorm.interval(mu,c(c1,cutoffs[k1,2])) + pnorm.interval(mu,c(cutoffs[k2,1], cc2))) -
                    const
    } else {
        - dnorm(cc2 - mu) + dnorm(c1 - mu) + mu * pnorm.interval(mu,c(c1,cc2)) - const
    }
}

## Compute g_mu(c1) for a vector of mu and c1 (see LaTeX documentation)
g.mu <- function(c1, mu, alpha, cutoffs,
                 pk=pnorm.cutoffs(mu, cutoffs), dk=dnorm.cutoffs(mu, cutoffs)) {
    sapply(1:length(c1),function(i)
           g.mu.single(c1[i], mu[i], alpha, cutoffs,
                       pk[i,,drop=FALSE], dk[i,,drop=FALSE]))
}

dnorm.cutoffs(c(0,0),two.sided.cutoff(10))
c.vals <- seq(-10.3,-10.28,.001)
plot(c.vals,g.mu(c.vals,rep(0,length(c.vals)), .05, two.sided.cutoff(10)))

g.mu(-10.2925, 0, .05, two.sided.cutoff(10))

## Compute g_mu'(c1)
dg.mu <- function(c1, mu, alpha, cutoffs,
                  pk=pnorm.cutoffs(mu, cutoffs)) {
    (c2(c1, mu, alpha, cutoffs, pk) - c1) * dnorm(c1 - mu)
}

points(c.vals, g.mu(c.vals-.001, rep(0,length(c.vals)), .05, two.sided.cutoff(10)) +
       dg.mu(c.vals-.001, rep(0,length(c.vals)), .05, two.sided.cutoff(10)) * .001,col="red",pch=3)

mu.vals <- seq(-10,15,.1)
plot(mu.vals, g.mu(rep(-10.2925,length(mu.vals)), mu.vals, .05, two.sided.cutoff(10)))
mu.vals <- seq(-.001,.001,.0001)
plot(mu.vals, g.mu(rep(-10.2925,length(mu.vals)), mu.vals, .05, two.sided.cutoff(10)))


## Compute upper CI endpoint, for a single x, when sigma=1
umau.normal.unit.var.upper.single <- function(x, cutoffs, alpha=.05, mu.lo=x, mu.hi=x+2, tol=1E-8) {
    check.cutoffs(cutoffs)

    mu.too.low <- function(mu) {
        g.mu(x,mu,alpha,cutoffs) > 0
    }
    mu.too.high <- function(mu) {
        g.mu(x,mu,alpha,cutoffs) < 0
    }

    while(mu.too.high(mu.lo)) {
        mu.hi <- mu.lo
        mu.lo <- mu.lo - 2
    }
    while(mu.too.low(mu.hi)) {
        mu.lo <- mu.hi
        mu.hi <- mu.hi + 2
    }
    while(mu.hi - mu.lo > tol) {
        mu.avg <- (mu.lo + mu.hi) / 2
        if(mu.too.high(mu.avg)) {
            mu.hi <- mu.avg
        } else {
            mu.lo <- mu.avg
        }
    }
    mu.avg
}

umau.normal.unit.var.upper.single(-10.29, two.sided.cutoff(10), mu.lo=-1, mu.hi=5)
umau.normal.unit.var.upper.single(-10.2925, two.sided.cutoff(10), mu.lo=-1, mu.hi=5)

## Compute both CI endpoints, for a single x
umau.normal.single <- function(x, cutoffs, sigma=1, alpha=.05, mu.lo=x, mu.hi=x+2, tol=1E-8) {
    mu.upper <- sigma * umau.normal.unit.var.upper.single(x/sigma, cutoffs/sigma, alpha,
                                                          mu.lo/sigma, mu.hi/sigma, tol)
    mu.lower <- -sigma * umau.normal.unit.var.upper.single(-x/sigma, negate.cutoffs(cutoffs)/sigma, alpha,
                                                           -mu.hi/sigma, -mu.lo/sigma, tol)
    return(c(mu.lower, mu.upper))
}

umau.normal.single(10.29, two.sided.cutoff(10))
umau.normal.single(1+10.29, 1+two.sided.cutoff(10))
umau.normal.single(-10.29, two.sided.cutoff(10))
umau.normal.single(1-10.29, 1+two.sided.cutoff(10))

umau.normal.single(10.005, two.sided.cutoff(10))

umau.normal.single(-13, two.sided.cutoff(10))


## Compute both CI endpoints, for a vector of x
umau.normal <- function(x, cutoffs, sigma=1, alpha=.05, tol=1E-8) {
    sapply(1:length(x), function(i) umau.normal.single(x[i], cutoffs, sigma, alpha, tol=tol))
}


## Make UMAU CIs for two different S's:
##   first,  (-Inf, -10) U (10, Inf)
##   second, (-Inf, -10) U (-0.1, 0.1) U (10, Inf)
## We see that even a small sliver of additional support keeps the
##  CIs from "reaching back" too far
x.vals <- c(seq(-20,-14,1),seq(-13,-10.4,.1),seq(-10.3,-10.06,.02),seq(-10.05,-10.01,.01),seq(-10.01,-10,.002))
length(x.vals)
CIs10 <- umau.normal(x.vals, two.sided.cutoff(10))


CIsMid <- umau.normal(x.vals, rbind(c(-Inf,-10),c(-.1,.1),c(10,Inf)))

pdf("UMAU.pdf",width=10)
par(mfrow=c(1,2))
matplot(x.vals,t(CIs10),type="l",xlim=c(-20,20),ylim=c(-23,23),
        main=expression(S==(-infinity*","*10)~U~(10*","*infinity)))
matplot(-x.vals,t(-CIs10),type="l",add=TRUE)
abline(h=0,lty=2)
abline(0,1,lty=3)

matplot(x.vals,t(CIsMid),type="l",xlim=c(-20,20),ylim=c(-23,23),
        main=expression(S==(-infinity*","*10)~U~(-.1*","*.1)~U~(10*","*infinity)))
matplot(-x.vals,-t(CIsMid),type="l",add=TRUE)
abline(h=0,lty=2)
abline(0,1,lty=3)
dev.off()

