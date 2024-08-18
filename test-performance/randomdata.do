**** Create Random Dataset for performance tests ****
*	
*	Performance tests use the three different sizes of data: small, medium, and large
*	Performance measured by how quickly each regression finishes (average) for within-regression comparison.
*	Total time for process completion measures time saved by parallelization

** Note: Keep in mind there are size limits depending on your edition of stata. 
** All shound comfortably be able to load all of these datasets without problem.  
** 


clear
set seed 42 // The Meaning of Life, the Universe, Everything. (Hitchhiker's Guide to the Galaxy, Ch 27) 




foreach size in 500000 5000000 500000000 1500000000{
	
	set obs `size'
	
	
	* Generate three independent variables (X1, X2, X3)
	gen X1 = rnormal()
	gen X2 = rnormal()
	gen X3 = rnormal()
	
	* Generate a binary dependent variable for logistic regression
	gen Y_bin = rbinomial(1, invlogit(0.5*X1 + 0.3*X2 - 0.2*X3))
	
	* Generate a count dependent variable for Poisson regression
	gen Y_count = rpoisson(exp(0.5*X1 + 0.3*X2 + 0.2*X3))

	* Generate a continuous dependent variable for linear regression
	gen Y_lin = 0.5*X1 + 0.3*X2 + 0.2*X3 + rnormal()

	* Generate high fixed effects
	gen fixed_effect = runiformint(1, 1000)
	
	if `size'==500000{
		save small, replace
	}
	if `size' == 5000000{
		save medium, replace
	}
	if `size' == 500000000{
		save large, replace
	}
	if `size'== 1500000000{
		save gigantic, replace
	}
	clear 
}




