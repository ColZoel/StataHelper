
# StataHelper
## A Simplified Python wrapper and Parallelization Library for Pystata

### Table of Contents
- [StataHelper](#StataHelper)
  - [Installation](#Installation)
  - [Introduction](#Introduction)
  - [Use Case: Looped Regressions](#Use-Case-Looped-Regressions)
  - [Usage](#Usage)
    - [Unified Interface](#Unified-Interface)
    - [Parallelization](#Parallelization)
  - [Documentation](#Documentation)
    - [Class: StataHelper](#Class-StataHelper)
    - [Methods](#Methods)
- [Contributing](#Contributing)
- [License](#License)
- [Author](#Author)
## Installation
There are two ways to install the package. The first is to install the package from PyPi using pip. 
The second is to clone the repository.

### From PyPi
```bash
pip install StataHelper
```
### From GitHub
```bash 
pip install git+
git clone 
cd StataHelper
pip install .
```
### Dependencies
- Python 3.4+
- Stata 16+ (Pystata is shipped with Stata licenses starting Stata 16)
- Pandas
- Numpy


## Introduction
Stata is a powerful package that boasts an impressive array of statistical tools, data manipulation capabilities,
and a user-friendly interface. Stata 16 extended its capabilities by introducing a Python interface, Pystata.
Intended especially for those with minimal Python experience, StataHelper is a Python wrapper around Pystata that does
the following:
- Simplifies the interface to interact with Stata through Python
- Provides a simple interface to parallelize Stata code
- Reads and writes data that cannot be imported directly to Stata, like Apache Parquet files

Note that parallelization in this case is not the same as multithreading as we see in Stata's 
off-the-shelf parallelization like Stata MP. In these cases, _calculations_ used in a single process (a regression, 
summary statistics, etc.) are passed through multiple cores. In contrast, StataHelper parallelization is used to run
multiple _processes_ across multiple cores simultaneously while still taking advantage of Stata's multithreading capabilities.

### Use Case: Looped Regressions

Suppose you have a set of regressions you want to run in which you change the dependent variable,
independent variables, or control variables. In Stata this would require several foreach-loops over the variables to 
change.

```stata
local ys depvar1 depvar2 depvar3
local xs indepvar1 indepvar2 indepvar3
local controls controlvar1 controlvar2 controlvar3

foreach y in local ys{
    foreach x in local xs {
        foreach control in local contros {
            regress `y' `x' `control'
            eststo model_`y'_`x'_`control'
        }
    }
}
```
Regression groups like this are common to identify the best model specification, especially in identifying how well a 
result holds across subsamples, fixed-effect levels, or time periods. Stata is a powerful tool
for this type of analysis, but is only designed to run a single regression at a time. 

For the sake of argument, let's say that Stata takes X seconds to run a single regression within any combination of parameters.
If we have 3 dependent variables, 3 independent variables, and 3 control variables, we would need to run 27 regressions.
This would take 27X seconds to run. 

Let's say we want to see if the result holds for two segments of the population
(e.g. heterogeneous effects), so now we have 3 dependent variables, 3 independent variables, 3 control variables, 
and 2 segments = 54 regressions, and an additional foreach-loop. This would take 54X seconds to run. 
As the number of variations increases, the time to run the regressions increases exponentially, 
each forloop has time complexity O(n), so the total time complexity is O(n^4).

This inefficiency is where StataHelper comes in. 

```python
from StataHelper import StataHelper
path = "C:/Program Files/Stata17/utilties"
s = StataHelper(stata_path=path, splash=False)
results = s.parallel("reg {y} {x} {control}", {'y': ['depvar1', 'depvar2', 'depvar3'],
                                               'x': ['indepvar1', 'indepvar2', 'indepvar3'],
                                               'control': ['controlvar1', 'controlvar2', 'controlvar3']})

```
The idea of parallelization is that we divide the number of regressions into smaller ques and run them simultaneously across
multiple cores. This reduces the time to run the regressions. If you have those 27 regressions and divide them evenly 
across 3 cores, you would reduce the time to run the regressions by 3X.

Additionally, StataHelper provides users a simplified interface to interact with pystata, can read and write data that 
cannot be imported directly to StataHelper, like Apache Parquet files, and can run StataHelper code from a string or a file.

# Usage


## Unified Interface
You can interact with StataHelper in nearly the same way you would interact with pystata. In pystata you would configure the
pystata instance as follows (assuming you have not added Stata to your PYTHONPATH):

```python
import sys

stata_path = "C:/Program Files/Stata17/utilties"
sys.path.append(stata_path)

from pystata import config

config.init(edition='mp', splash=False)
config.set_graph_format('svg')
config.set_graph_size(800, 600)
config.set_graph_show(False)
config.set_command_show(False)
config.set_autocompletion(False)
config.set_streaming_output(False)
config.set_output_file('output.log')

from pystata import stata

stata.run('di "hello world"')
stata.run("use data.dta")
stata.run("reg y x")
config.close_output_file()  # Close the Stata log
```

Notice how we have to configure the stata instance before we can even call import the `stata` module, 
and the stata instance requires a separate `config` object to be configured. 

In StataHelper, you can configure the Stata instance directly in the constructor.

```python
from StataHelper import StataHelper

s = StataHelper(splash=False,
                    edition='mp',
                    set_graph_format='svg',
                    set_graph_size=(800, 600),
                    set_graph_show=False,
                    set_command_show=False,
                    set_autocompletion=False,
                    set_streaming_output=False,
                    set_output_file='output.log')
s.run("di hello world")
s.run("use data.dta")
s.run("reg y x")
s.close_output_file()
```


## Parallelization
StataHelper provides a simple interface to parallelize StataHelper code. Just as with pystata's `run` method, 
you may pass a string of StataHelper code to the `parallel` method. StataHelper is designed to read placeholders in the stata
code for the values you wish to iterate over. There are two methods to do this:


### Brace Notation
The previous snippet exemplifies brace notation, which is intended to be intuitive. All this is needed is the command, and a dictionary with the keys 
as the placeholders. The values can be any iterable object.

```python
parameters = {'control': ['controlvar1', 'controlvar2', 'controlvar3'],
              'x':['indepvar1', 'indepvar2', 'indepvar3'], 
              'y': ['depvar1', 'depvar2', 'depvar3']}
```
Dictionaries are inherently order-agnostic, so the order of the keys does not matter as long as all keys are in the command 
and all placeholders in the command are keys in the dictionary. The order of the keys will only 
affect the unique identifier of the results in the output directory (see below).


### Multi-level Parameters
Let's say you wanted to run a series of regressions but vary the level of fixed effects. You would approach this by 
simply introducing a sublist into the fixed effects list. In the following example, we'll use the `reghdfe` command to
run a series of regressions with varying high-dimensional fixed effects.

```python
from StataHelper import StataHelper

values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'fixed_effects': [['fe1', 'fe2'], ['fe1'], ['fe1', 'fe5']]}
s = StataHelper(splash=False)
s.run("ssc install reghdfe")
results = s.parallel("reghdfe {y} {x} absorb({fixed_effects})", values)
```

### Multiline Stata Code
You can pass multiline StataHelper code to the `parallel` method just as you would with `pystata.stata.run`.

```python
import StataHelper

stata = StataHelper.StataHelper(splash=False)
values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'control': ['controlvar1', 'controlvar2', 'controlvar3']}

results = stata.parallel("""
               reg {y} {x} {control}
               predict yhat
               gen residuals = {y} - yhat
               """, values)
```

### Conditional Statements
You can also pass conditional statements to the `parallel` method to analyze a subset of the data.

```python
import StataHelper

stata = StataHelper.StataHelper(splash=False)
values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'control': ['controlvar1', 'controlvar2', 'controlvar3'],
          'subsets': ['var4<=2023 & var5==1', 'var4>2023 | var5==0']}

results = stata.parallel("reg {y} {x} {control} if {subsets}", values)
```

### Saving Estimation Results
Estimation commands can be saved to a file by specifying the `est save` command in `cmd`. 
The `parallel` method will save the results to a folder titled `name` in `set_output_dir` if `name` is not `None` and 
an asterisk `*` is present in `cmd`.
All files in this directory are also called `name`, but with a unique identifier appended to the end. 
e.g.
```python
from StataHelper import StataHelper
s = StataHelper(edition='mp', splash=False, set_output_dir='C:/Users/me/Documents/StataOutput')

...

s.parallel("eststo: reg y {x} if {subset}\nest save *", values, name='regressions')
```
produces the following files in `C:/Users/me/Documents/StataOutput/regressions`:
```
regressions_1.ster
regressions_2.ster
regressions_3.ster
regressions_4.ster
regressions_5.ster
regressions_6.ster
```
You can easily load these files into Stata by looping over the files in the directory and using the `est restore` command.
After which, you can use the `esttab` command to create a table of the results just as if you had run the regressions in a
loop in Stata.


In general, this method can be used to run many types of stata commands in parallel, not just regressions.
You might, for example, want to run a series of `tabstat` commands to summarize the data and save the results 
to a file. 

```python 
from StataHelper import StataHelper
s = StataHelper(edition='mp', splash=False, set_output_dir='C:/Users/me/Documents/StataOutput')

values = {'var': ['var1', 'var2', 'var3', 'var4', 'var5']}
s.parallel("tabstat {var}, stat(mean sd) save *.xlsx", values, name='table1')
```
produces the following files in `C:/Users/me/Documents/StataOutput/table1`:
```
table1_1.xlsx
table1_2.xlsx
table1_3.xlsx
table1_4.xlsx
table1_5.xlsx
```


# Documentation

[//]: # (<style>)

[//]: # (ul {)

[//]: # (    list-style-type: none;)

[//]: # (})

[//]: # (</style>)

<!-- ************* WARNING ************  -->

<p><span style="color:red; border:0 solid red; padding:5px;">Note: Wrappers and arguments for StataNow functionalities have not been tested. They are included for completeness. 
See Pystata documentation for more information. See below for information about contributing to the project.</span></p>



<!-- ************* StataHelper Class ************  -->

## Class: StataHelper
<h3>StataHelper(_self, edition=None, splash=None, set_output_dir=None, set_graph_format=None, set_graph_size=None, set_graph_show=None, 
set_command_show=None, set_autocompletion=None, set_streaming_output=None, set_output_file=None</em> </p></h3>
<ul>
<li><strong>edition</strong><em>  (str)</em> :
The edition of Stata to use.</li>
<li><strong>splash</strong><em>  (bool)</em> :  Whether to show the splash screen when StataHelper is opened.
    It is recommended not use this when running parallel, as it will repeat for every core that is opened.</li>
<li><strong>set_output_dir</strong><em>  (str)</em> : 
    The directory to save the output files such as estimation files. A new folder housing these files is created in this directory.</li>
<li><strong>set_graph_format</strong><em>   (str)</em> : pystata.config.set_graph_format. 
    The format of the graphs to be saved.</li>
<li><strong>set_graph_size</strong><em>  (tup)</em> : pystata.config.set_graph_size. The size of the graphs to be saved.</li>
<li><strong>set_graph_show</strong><em>  (bool)</em> : pystata.config.set_graph_show. 
    Whether to show the graphs in the StataHelper window.</li>
<li><strong>set_command_show</strong><em>  (bool)</em> : pystata.config.set_command_show. 
    Whether to show the commands in the StataHelper window.</li>
<li><strong>set_autocompletion</strong><em>  (bool)</em> : pystata.config.set_autocompletion. 
    Whether to use autocompletion in the StataHelper window.</li>
<li><strong>set_streaming_output</strong>: pystata.config.set_streaming_output. 
    Whether to stream the output to the console.</li>
<li><strong>set_output_file</strong><em>  (str)</em> : pystata.config.set_output_file. 
    Where to save the Stata log file.</li>
</ul>

All values not specified as an argument default to the pystata defaults. See the
[pystata documentation](https://www.stata.com/python/pystata18/config.html).

## Methods

<h3>StataHelper.is_stata_initialized(_self_)</h3>
<ul>
<li>Wrapper for <code>pystata.stata.is_initialized()</code>.</li><br>
<li>Returns True if Stata is initialized, False otherwise.</li>
</ul>


<h3>StataHelper.status(_self_)</h3>
<ul>
<li>Wrapper for <code>pystata.stata.status()</code>.</li><br>
<li>Prints the status of the Stata instance to the console. Returns None.</li>
</ul>

<h3>StataHelper.close_output_file(_self_)</h3>
<ul>
<li>Wrapper for <code>pystata.stata.close_output_file()</code>.</li><br>  
<li>Closes the Stata log file.</li>
</ul>

### **StataHelper.get_return**(_self_)

<ul>
<li>Wrapper for <code>pystata.stata.get_return()</code>.</li><br>
<li>Returns the <code>return</code> values from the last Stata command as a dictionary.</li>
</ul>

### **StataHelper.get_ereturn**(_self_)
<ul>
<li>Wrapper for <code>pystata.stata.get_ereturn()</code>.</li><br>
<li>Returns the <code>e(return)</code> values from the last Stata command as a dictionary.</li>
</ul>


### **StataHelper.get_sreturn**(_self_)
<ul>
<li>Wrapper for <code>pystata.stata.get_sreturn()</code>.</li><br>
<li>Returns the <code>sreturn</code> values from the last Stata command as a dictionary.</li>
</ul>


### **StataHelper.run**(_self, cmd, **kwargs_)
<ul>
<li>Wrapper for <code>pystata.stata.run()</code>.</li><br>
<li>Runs cmd in the Stata window.</li>
<li><strong>cmd</strong><em>  str</em> : Stata command to run.</li>
<li><strong>**kwargs</strong><em>  dict</em> : Additional arguments to pass to the Stata command.</li>
</ul>

### **StataHelper.use(_self, data, columns=None, obs=None, \*\*kwargs_)**
<ul>
<li>Pythonic method to load a dataset into Stata. Equivalent to <code>use</code> command in Stata.</li><br>
<li><strong>data</strong><em>  str</em> : The path to the data file to load into Stata.</li>
<li><strong>columns</strong><em>  list or str</em> : The columns to load into Stata. If None, all columns are loaded.</li>
<li><strong>obs</strong><em>  int or str</em> : The number of observations to load into Stata. If None, all observations are loaded.</li>
</ul>

### **StataHelper.use_file(_self, path, frame=None, force=False, \*args, \*\*kwargs_)**
<ul>
<li>Read any pandas-supported file into Stata. Equivalent to <code>import delimited</code> in Stata for delimited files.</li>
<li>This method allows some files that cannot be imported directly into Stata to be loaded.</li>
<br>
<li><strong>path</strong><em>  str</em> : The path to the file to load into Stata.</li>
<li><strong>frame</strong><em>  str</em> : The name of the frame to load the data into. If None, the file name is used.</li>
<li><strong>force</strong><em>  bool</em> : Whether to overwrite the existing frame. If False, the frame is appended.</li>
<li>Raises a <code>ValueError</code> if the extension is not in the list of supported file types.</li>
<br>
<li>Valid file types include: CSV, Excel, Parquet, Stata, Feather, SAS, SPSS, SQL, HTML, JSON, pickle/compressed files, xml, clipboard.</li>
</ul>


### StataHelper.use_as_pandas(_self, frame=None, var=None, obs=None, selectvar=None, valuelabels=None, missinglabels=\_DefaultMissing(), \*args, \*\*kwargs_)**
<ul>
<li>Read a Stata frame into a Pandas DataFrame. Equivalent to <code>export delimited</code> in Stata for delimited files.</li><br>
<li><strong>frame</strong><em>  str</em> : The name of the frame to read into a pandas DataFrame. If None, the active frame is used.</li>
<li><strong>var</strong><em>  list or str</em> : The variables to read into the DataFrame. If None, all variables are read.</li>
<li><strong>obs</strong><em>  int or str</em> : The number of observations to read into the DataFrame. If None, all observations are read.</li>
<li><strong>selectvar</strong><em>  str</em> : The variable to use as the index. If None, the index is not set.</li>
<li><strong>valuelabels</strong><em>  bool</em> : Whether to use value labels. If True, the value labels are used. If False, the raw values are used.</li>
<li><strong>missinglabels</strong><em>  str</em> : The missing value labels to use. If None, the default missing value labels are used.</li>


This method allows some files that stata cannot export directly to be read into a pandas DataFrame. 
In the case of .dta files, this method is significantly faster than using the <code>pandas.read_stata</code> method as the dataset 
is first loaded into Stata and then read into a Pandas DataFrame. 
</ul>

### **StataHelper.save(_path, frame=None, var=None, obs=None, selectvar=None, valuelabel=None, missinglabel=None, missval=\_DefaultMissing(), \*args, \*\*kwargs_)**
<ul>
<li>Save a Stata dataset to a file. Passes the frame to Pandas and saves the file using the Pandas method. Valid file types are the same as <code>use_file</code>.</li> <br>
<li><strong>path</strong><em>  str</em> : The path to save the file to.</li>
<li><strong>frame</strong><em>  str</em> : The name of the frame to save. If None, the active frame is used.</li>
<li><strong>var</strong><em>  list or str</em> : The variables to save. If None, all variables are saved.</li>
<li><strong>obs</strong><em>  int or str</em> : The number of observations to save. If None, all observations are saved.</li>
<li><strong>selectvar</strong><em>  str</em> : The variable to use as the index. If None, the index is not set.</li>
<li><strong>valuelabels</strong><em>  bool</em> : Whether to use value labels. If True, the value labels are used. If False, the raw values are used.</li>
<li><strong>missinglabels</strong><em>  str</em> : The missing value labels to use. If None, the default missing value labels are used.</li>
<li><strong>missval</strong><em>  str</em> : The missing value labels to use. If None, the default missing value labels are used.</li><br>
<li>Raises a <code>ValueError</code> if the extension is not in the list of supported file types.</li>
<br>
<li>Valid file types include: CSV, Excel, Parquet, Stata, Feather, SAS, SPSS, SQL, HTML, JSON, pickle/compressed files, xml, clipboard.</li>
</ul>
</ul>

### **StataHelper.schedule(_self, cmd, pmap_)**
<ul>
<li>Returns the queue of commands to be run in parallel (cartesian product). Analogous to the parallel method, but does not execute the commands.</li>
<li><strong>cmd</strong><em>  str</em> : The Stata command to run in parallel.</li>
<li><strong>pmap</strong><em>  dict</em> : The parameters to iterate over in the Stata command. Can be any iterable object of any dimension, but note that the deeper the dimension, the more (potentially redundant) combinations are created.</li>
<br><li>All keys in pmap must be in cmd, and all placeholders in cmd must be in pmap.</li>
</ul>
This method creates a queue of string commands to be run in parallel by replacing the bracketed values with their respective values in the cartesian product of the values in pmap.
"Queue" is used loosely here, as the commands are not run sequentially and there is no guarantee of the order in which they are run in parallel.
However, each process's command is labelled with a unique identifier in the order of the queue.


For example, 
```python
from StataHelper import StataHelper
s = StataHelper(splash=False)
values = {'x': [['indepvar1', 'indepvar2'], 'indepvar1', 'indepvar2', 'indepvar3']}
s.schedule("reg y {x}", values)
```
would place the following regressions in queue.
```stata
reg y indepvar1 indepvar2
reg y indepvar1
reg y indepvar2
reg y indepvar3
```
Values can also be conditional statements.

```python
from StataHelper import StataHelper
s = StataHelper(splash=False)
values = {'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'subset': ['var1==1', 'var2==2', 'var3==3']}
s.schedule("reg y {x} if {subset}", values)
```
returns the following regressions in queue
```stata
reg y indepvar1 if var1==1
reg y indepvar1 if var2==2
reg y indepvar1 if var3==3
reg y indepvar2 if var1==1
reg y indepvar2 if var2==2
reg y indepvar2 if var3==3
reg y indepvar3 if var1==1
reg y indepvar3 if var2==2
reg y indepvar3 if var3==3
```
Logical operators can be specified in the conditional statement.

```python
from StataHelper import StataHelper
s = StataHelper(splash=False)
values = {'x': ['indepvar1', 'indepvar2'],
          'subset': ['var1==1 & var2==2', 'var2==2 | var3==3', 'var3==3']}
s.schedule("reg y {x} if {subset}", values)
```
returns:
```stata
reg y indepvar1 if var1==1 & var2==2
reg y indepvar1 if var2==2 | var3==3
reg y indepvar1 if var3==3
reg y indepvar2 if var1==1 & var2==2
reg y indepvar2 if var2==2 | var3==3
reg y indepvar2 if var3==3
```


### **StataHelper.parallel**(_self, cmd, pmap, name=None, max_cores=None, safety_buffer=1)
<ul>
<li>Runs a series of Stata commands in parallel. Analogous to the schedule method, but executes the commands.</li>
<li><strong>cmd</strong><em>   str</em> : The StataHelper code to run in parallel, including placeholders for the values to iterate over. Placeholders use brace notation <code>{}</code>.
<code>pmap</code> must be a dictionary with keys that match the placeholders in the StataHelper code.</li>
<li><strong>pmap</strong><em>   list, dict, tuple</em> : The values to iterate over in the StataHelper code. If a list or tuple, the order of the values. If a dict, the order only matters if you use wildcards. In that case, the keys are ignored. Items in sublists are joined with a whitespace <code>" "</code> and allow multiple values for a single placeholder.</li>
<li><strong>name</strong><em>  str</em> : The name of the output directory, replacing <code>*</code> in <code>cmd</code>. If None, a unique identifier is created based on the order of the process in the queue.</li>
<li><strong>max_cores</strong><em>  int</em> : The maximum number of cores to use. If <code>None</code>, then the <code>min(os.cpus()-safety_buffer, len(pmap))</code> is used. if <code>max_cores</code> is greater than the number of combinations and the number of combinations is greater than the number of cores, then <code>os.cpus()-safety_buffer</code> are used.</li>
<li><strong>safety_buffer</strong><em>  int</em> : The number of cores to leave open for other processes.</li>
</ul>




---
# Contributing
Contributions are welcome! If you would like to contribute to the project, 
please fork the repository and submit a pull request. Specifically, we are looking for contributions in the following areas:
- Testing current functionalities on multiple platforms
- Testing and following StataNow functionalities
- within-Stata multiprocessing (to migrate away from the `multiprocessing` module)
- Applications of NLP or LLM in troubleshooting stata errors and summarizing help files 

# License


# Author
Collin Zoeller, Tepper School of Business, Carnegie Mellon University
<br>[zoellercollin@gmail.com](mailto:zoellercollin@gmail.com)
<br>Github: [ColZoel](https://github.com/ColZoel)
<br>Website: [colzoel.github.io](https://colzoel.github.io)


---
_Author Collin Zoeller and StataHelper are not affiliated with StataCorp. Stata is a registered trademark of StataCorp 
LLC. While StataHelper is open source, Stata and its Python API Pystata are proprietary software and require a license. 
See [stata.com](https://www.stata.com/) for more information._

###### tags: `Stata` `Python` `Pystata` `Parallelization` `StataHelper` `Documentation`
###### August 2024
