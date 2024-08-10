*** Test Combination Regression in GUI***
ssc install reghdfe // for high-dimensional fixed effects 
timer clear
cd "D:\Collin\PyCharmProject\Pyata\test-performance"

* System info 
local stata_version = c(stata_version)
local stata_flavor  = c(flavor)
local os 			= c(os)
local machine 		= c(machine_type)
local cpus_stata 	= c(processors)
local cpus_mach 	= c(processors mach)


local reps 100

* Initialize variables for each regression
local lin_no_fe_total 	0
local lin_fe_total 		0
local poisson_total 	0
local logistic_total 	0

for each size in small medium large gigantic{
	use `size', clear
	
	forval i = 1/`reps' {
		
		* Linear regression without fixed effects
		timer clear
		timer on 1
		regress Y_lin X1 X2 X3
		timer off 1
		local lin_no_fe_total = `lin_no_fe_total' + `r(t1)'
		
		* Linear regression with fixed effects
		timer clear
		timer on 2
		reghdfe Y_lin X1 X2 X3, absorb(fixedeffect)
		timer off 2
		local lin_fe_total = `lin_fe_total' + `r(t2)'
		
		* Poisson regression
		timer clear
		timer on 3
		poisson Y_count X1 X2 X3
		timer off 3
		local poisson_total = `poisson_total' + `r(t3)'
		
		* Logistic regression
		timer clear
		timer on 4
		logit Y_bin X1 X2 X3
		timer off 4
		local logistic_total = `logistic_total' + `r(t4)'
	}

	* Calculate the average times 
	local lin_no_fe_avg = `lin_no_fe_total' / `reps'
	local lin_fe_avg = `lin_fe_total' / `reps'
	local poisson_avg = `poisson_total' / `reps'
	local logistic_avg = `logistic_total' / `reps'
	local total_avg = `lin_no_fe_avg' + `lin_fe_avg' + `poisson_avg' + `logistic_avg'

	* Create a new dataset to store the average times
	clear
	input str30 regression_type avg_time
	
	"Stata version"  				`stata_version'		
	"Stata Edition"  				`stata_flavor'
	"os" 			 				`os'
	"machine" 		 				`machine'
	"Stata CPUs" 	 				`cpus_stata'
	"Machine CPUs" 	 				`cpus_mach'
	"Number of Loops" 				`reps'
	"Linear Regression (No FE)" 	`lin_no_fe_avg'
	"Linear Regression (With FE)" 	`lin_fe_avg'
	"Poisson Regression" 			`poisson_avg'
	"Logistic Regression" 			`logistic_avg'
	"Average Process Time" 			`total_avg'
	end

	* Export the results to a CSV file
	export delimited using "avg_time_`size'_gui.csv", replace
}