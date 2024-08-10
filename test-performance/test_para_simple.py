"""
Test The simple regressions in parallel. Each regression is run in a separate thread
"""
from StataHelper import StataHelper
import os
s = StataHelper(stata_path=r"C:\Program Files\Stata18\utilities", edition='mp', splash=True)
os.chdir(os.path.dirname(__file__))

systeminfo = """ local stata_version = c(stata_version)
                 local stata_flavor  = c(flavor)
                 local os 			= c(os)
                 local machine 		= c(machine_type)
                 local cpus_stata 	= c(processors)           
                 local cpus_mach 	= c(processors mach)"""


# Can use the same time id because the commands are run in separate stata instances

cmd = """
        local stata_version = c(stata_version)
         local stata_flavor  = c(flavor)
         local os 			= c(os)
         local machine 		= c(machine_type)
         local cpus_stata 	= c(processors)           
         local cpus_mach 	= c(processors mach)
    
        use {size}.dta, clear
        forval i = 1/{reps} {{
        local time_total 0
        timer clear
		timer on 0
		{{func}} X1 X2 X3
		timer off 0
		local time_total = `time_total' + `r(0)'
		}}
		local avg_time = `time_total'/100
		
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
        "{func}" 						`avg_time'
        end
    
    * Export the results to a CSV file
	export delimited using "avg_time_para_{size}_{reps}.csv", replace
		"""

params = {'func': ['regress Y_lin', 'reghdfe Y_lin', 'poisson Y_count', 'logit Y_bin'],
          'reps': [100],
          'size': ["small", "medium", "large", "gigantic"]}

s.parallel(cmd, params, quietly=False)