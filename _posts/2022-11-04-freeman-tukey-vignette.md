<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

---
Testing goodness-of-fit in spatial capture-recapture models with the Freeman-Tukey test
---

# The Freeman-Tukey goodness-of-fit test
The Freeman-Tukey test is a goodness-of-fit (GoF) test for models of count data. Given a set of $B$ observations, the Freeman-Tukey statistic ($FT$) quantifies the discrepancy between the observed counts ($O$) and the expected counts ($E$), summed across each observation $b$:
$$ FT = \displaystyle \sum\limits_{b=1}^B \left( \sqrt{O_b} - \sqrt{E_b} \right)^2 $$
A larger value for $FT$ would thus indicate a greater discrepancy between the observed data and expected values estimated by the model. 

Since a model does not fit data perfectly, the Freeman-Tukey statistic will always be greater than zero. To infer if the model provided a good fit to the data,
1. Simulate data from the parameters estimated by the model
    - parametric bootstrapping for frequentist models
    - posterior predictive checks (PPC) for Bayesian models
2. Calculate the Freeman-Tukey statistic for each simulated data set
3. Compare the (distribution of) the Freeman-Tukey statistic(s) for the observed data ($FT_{obs}$) against the distribution for the simulated data ($FT_{sim}$)

For a model that had provided a good fit to the data, $FT_{obs}$ should lie near the centre of the distribution of $FT_{sim}$ for frequentist models, or display no systematic differences from the distribution of $FT_{sim}$ for Bayesian models. 

# Application of Freeman-Tukey GoF test in SCR models
SCR data $y_{ijk}$ records the capture histories of individual $i$ at trap $j$ on occasion $k$ and usually takes the form of binary data, for which fit or lackthereof is difficult to assess. It is therefore more useful to aggregate the data along one or more dimensions, to obtain summary statistics of the remaining dimensions, eg. aggregating across $K$ sampling occasions, such that the data $y_{ij.}$ summarises the encounter frequency of each individual at each trap. The aggregated data subsequently takes the form of count data, with which GoF can be assessed using the Freeman-Tukey GoF test.

## Assessing GoF based on $y_{ij.}$
In using $y_{ij.}$ for testing GoF, we test if the data is consistent with expected individual patterns in capture frequencies at each trap given the trap's distance to the individual's activity centre. The Freeman-Tukey statistic is calculated by
$$FT1 = \displaystyle\sum\limits_{i=1}^n \displaystyle\sum\limits_{j=1}^J \left(\displaystyle\sqrt{y_{ij.}} - \displaystyle \sqrt{E(y_{ij.})} \right) ^2 \\
\mathrm{where} \,\, E(y_{ij.}) = \sum\limits_{k=1}^K p_{ijk}$$
with $p_{ijk}$ being the probability of capturing individual $i$ at trap $j$ on occasion $k$.
The test may be indicative of poor fit due to
- a misspecified detection function 
- unmodelled environmental and trap heterogeneity in detection parameters 
- unmodelled individual heterogeneity if sex ratios are skewed
- unmodelled heterogeneity or misspecified model for density  

Since the data aggregates across sampling occasions, GoF tests with $y_{ij.}$ do not allow us to infer if there may have been unmodelled temporal heterogeneity in detection.

# Testing GoF in practice with frequentist SCR models
In frequentist SCR models, the activity centres ($\bm{s}$) are not explicitly estimated as a model parameter but rather integrated across the state-space to estimate density. Since $p_{ijk}$ which depends on the location of $s_i$, we derive the posterior probability of $s_i$ in the $\pi(s_i)$ given the model estimates $\bm{\hat{\theta}}$ and the animal's capture history $y_i$. Using Bayes rule,
$$\left[ s_i \mid y_i,\, \bm{\hat{\theta}} \right]
  \propto \left[ y_i,\, \bm{\hat{\theta}} \mid s_i \right] \left[ s_i \right] \\
  \implies \pi(s_i) = \left[s_i \mid y_i,\, \bm{\hat{\theta}} \right] \\
	= \dfrac{\left[ y_i,\, \bm{\hat{\theta}} \mid s_i \right] \left[ s_i \right]} {\displaystyle\int \left[ y_i,\, \bm{\hat{\theta}} \mid s_i \right] \left[ s_i \right] d \mathcal{S}}$$
In practice, $\mathcal{S}$ is discretised into a mask with $G$ pixels to allow the fitting of SCR models to be computationally feasible. In deriving the locations of $\bm{s}$, we can use the mask to calculate to calculate $\pi(s_{ig})$, the probability that $s_i$ is located in pixel $\mathcal{S}_g$ and $\displaystyle\int \left[ y_i,\, \bm{\hat{\theta}} \mid s_i \right] \left[ s_i \right] d \mathcal{S}$ is approximated by summing across the likelihoods that $s_i=\mathcal{S}_g$. Deriving $\pi(s_i)$ is part of the  ```GoF_HN``` function
```{R}
	# Extract predicted detection parameters
	preds <- predict(object)
	N_bs <- ceiling(preds[1,2] * spatstat.geom::area(object$mask) / 1e4)
	g0_bs <- preds[2,2]
	sigma_bs <- preds[3,2]

  # Function to calculate Euclidean distance between coords and traps
  eucDist <- function(xy1, xy2){
    i <- sort(rep(1:nrow(xy2), nrow(xy1)))
    dvec <- sqrt((xy1[, 1] - xy2[i, 1])^2 + (xy1[, 2] - xy2[i, 2])^2)
    matrix(dvec, nrow=nrow(xy1), ncol=nrow(xy2), byrow=F)
  }
	# Distances between each s and trap
	dmat <- eucDist(s, traps)
	# Probability of capture at each trap for each s, assuming HN detection fn
	pmat <- g0_bs * exp(-dmat^2 / (2 * sigma_bs^2))
	# Expected capture history given each s
	emat <- pmat * nOcc

	
	# Get probability of all [s | y]
  # Include [s | y] for one uncaptured individual
  if(!is.null(pi_S)){
     piS <- pi_S
  } else {
    lik <- piS <- matrix(NA, nrow(s), nrow(Y) + 1)
    Y_all <- abind::abind(Y, 
      array(0, dim=c(1, dim(Y)[2:3])), along=1)
    # Unconditional probability [s] 
    ds <- rep(1/nrow(s), nrow(s))
    for(i in 1:nrow(Y_all)){
      for(g in 1:nrow(s)){
        # [s | y] = [y | s] * [s]
        # Calculate [s | y] using trap-specific p_ijk for each s
        lik[g, i] <- exp(sum(
          dbinom(colSums(Y_all[i,,]), nOcc, pmat[g,], log=TRUE))) * ds[g]
      }
      piS[,i] <- lik[,i] / sum(lik[,i])
    } 
  }
   
  # Convert s posterior to image for drawing points
  s_im <- as.im(as.data.frame(cbind(s, piS)))
```

To account for the uncertainty of $s_i$ in $\pi(s_i)$, in each bootstrap resample, we simulate a realisation of $s_i$ from $\pi(s_i)$ and calculate $\hat{p_{ijk}}$ from the detection parameters $\hat{g_0}$ and $\hat{\sigma}$, and the distance between each trap and $s_i$, thus resulting in a new value of $E(y_{ij.})$ for each resample. This produces a distribution of values for both $FT_{obs}$ and $FT_{sim}$ in a similar fashion to the Bayesian PPC. As with PPC, systematic differences between $FT_{obs}$ and $FT_{sim}$ indicate a lack of fit.

# Demonstration of GoF tests for SCR data in R
Here, we demonstrate how GoF may be carried out in practice using simulated data and the ```GoF_HN``` function
## Simulate the data
We first set up the packages needed in the environment to simulate SCR data and fit SCR models, as well as define the true model parameters
```{R}
# load packages for fitting SCR models
library(secr)
library(spatstat)
library(maptools)

# Set up SCR parameters
# Biological model
# Simulate activity centres
set.seed(4726)
N <- 50   # number of animals
win <- owin(c(-0.5, 50.5), c(-0.5, 50.5))  # window for activity centres
AC <- runifpoint(N, win=win) 
trueD <- N/area(win) * 1e4

nOcc <- 30  # no. of occasions
# Simulate detection parameters for an annular normal detection function
g0 <- 0.05  		# detection probability at location(s) of highest activity
sigma <- 4  # movement parameter
```
We then create a state-space and populate it with traps
```{R}
trapSpace <- 5 # trap spacing
# Create traps
traps <- expand.grid(x=seq(10, 40, trapSpace), y=seq(10, 40, trapSpace))
# Create state space
S <- expand.grid(x=seq(0, 50, 1), y=seq(0, 50, 1))

# Add class and attributes to traps for readibility in secr
class(traps) <- c("traps", "data.frame")
attr(traps, "detector") <- "proximity"
attr(traps, "spacex") <- trapSpace
attr(traps, "spacey") <- trapSpace
attr(traps, "spacing") <- trapSpace
```
SCR data is simulated using the ```simHN``` function which generates data assuming a half-normal detection function
```{R}
# Simulate SCR data
set.seed(324)
CH <- simHN(AC=AC, traps=traps, g0=g0, sigma=sigma, nOcc=nOcc)[[2]]
```
## Fitting the SCR model
The continuous state-space is discretised into an ```secr.mask``` object to calculate the integration of $\bm{s}$ across $\mathcal{S}$
```{R}
# Make SCR mask 
polywin <- as(win, "SpatialPolygons")
traps_secr <- read.traps(data=traps, detector="proximity")
popn <- data.frame(x=AC$x, y=AC$y)
class(popn) <- c("popn", "data.frame")
mask <- make.mask(traps, type="polygon", poly=polywin, spacing=1)
```
We then fit a SCR model assuming a half-normal detection function to the data using the ```secr``` package
```{R}
# Fit SCR model
fit1 <- secr.fit(CH, mask=mask, details=list(fastproximity=FALSE))
```
We used ```fastproximity=FALSE``` to avoid collapsing the $y_{ijk}$ array stored in the ```secr.fit``` object so that the model output can be used directly in ```GoF_HN```, though the GoF function could be modified to allow for both types of data structures.
## Testing GoF of the SCR model
The ```secr.fit``` object can be fed directly to ```GoF_HN```, which calculates GoF assuming a null model with a half-normal detection function.
```{R}
# Test goodness-of-fit of SCR model
# Check FT1 (Freeman-Tukey test for ind x trap encounters)
scrGoF_FT1 <- GoF_HN(fit1, test="FT1")
```
For the purposes of this exercise, we conduct only the Freeman-Tukey GoF test for $y_{ij.}$, though the function may also be used to carry out the Freeman-Tukey test on summed individual encounter frequencies $y_{i..}$ and summed trap encounter frequencies $y_{.j.}$, as well as the $\chi^2$ tests on these summary statistics.