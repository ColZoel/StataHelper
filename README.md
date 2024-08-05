
# StataHelper
## A Simplified Python wrapper and Parallelization Library for Pystata

### Table of Contents
- [Installation](#installation)
- [Introduction](#introduction)
- [Parallelization](#parallelization)
    - [Brace Notation](#brace-notation)
    - [Multi-level Parameters](#multi-level-fixed-effects)
    - [Multiline Stata Command](#multiline-stata-code)
    - [Conditional Statements](#conditional-statements)
- [Unified Interface](#unified-interface)
- [Documentation](#documentation)
- [Class: StataHelper](#class-stata)
    - [StataHelper.run](#statarunself-code)
    - [StataHelper.parallel](#stataparallelself-code-values-stata_wildcardfalse-max_coresnone-safety_buffer1-show_batchesfalse)
    
## Installation
There are two ways to install the package. The first is to install the package from PyPi using pip. 
The second is to clone the repository.

### From PyPi
```bash
pip install StataHelper
```

```bash 
pip install git+
git clone 
cd StataHelper
pip install .
```
### Dependencies
- Python 3.9+
- Stata 16+ (Pystata is shipped with most Stata licenses starting Stata 16)
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
stata = StataHelper(stata_path=path, splash=False)
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


```python
import StataHelper

stata = StataHelper.StataHelper(config='config.yaml')
stata.run("di hello world")
```

You can also get the same effect by passing a dictionary with the same keys to the constructor, which 
effectively how the YAML is parsed under the hood.

```python
import StataHelper

dict = {'path': "C:/Program Files/Stata17/StataMP-64.exe",
        'splash': False,
        'edition': 'mp',
        'set_graph_format': 'svg',
        'set_graph_size': [800, 600],
        'set_graph_show': False,
        'set_command_show': False,
        'set_autocompletion': False,
        'set_streaming_output': False
        }
stata = StataHelper.StataHelper(config=dict)
```
All values not specified either as an argument or in the YAML file default to the pystata defaults. See the
[pystata documentation](https://www.stata.com/python/pystata18/config.html).

# Documentation
<div style="color:red; border:0 solid red; padding:5px;">
Note: Wrappers for StataNow functionalities have not been tested. They are included for completeness. See pystata documentation for more information.
See below for information about contributing to the project.
</div>

## Class: StataHelper
 **StataHelper**(_self,
                 edition=None,
                 splash=None,
                 set_graph_format=None,
                 set_graph_size=None,
                 set_graph_show=None,
                 set_command_show=None,
                 set_autocompletion=None,
                 set_streaming_output=None,
                 set_output_file=None)_

**edition**_(str)_ : The edition of StataHelper to use. 

 **splash**_(bool)_:  Whether to show the splash screen when StataHelper is opened. It is recommended not use this when running parallel, 
as it will repeat for every core that is opened.
 
 **set_graph_format**_(str)_: <br>pystata.config.set_graph_format. The format of the graphs to be saved.

 **set_graph_size**_(tup)_: pystata.config.set_graph_size. The size of the graphs to be saved.

 **set_graph_show**_(bool)_: pystata.config.set_graph_show. Whether to show the graphs in the StataHelper window.

 **set_command_show**_(bool)_: pystata.config.set_command_show. Whether to show the commands in the StataHelper window.

 **set_autocompletion**_(bool)_: pystata.config.set_autocompletion. Whether to use autocompletion in the StataHelper window.

 **set_streaming_output**: pystata.config.set_streaming_output. Whether to stream the output to the console.

 **set_output_file**_(str)_: pystata.config.set_output_file. Where to save the Stata log file.

## Methods


### **StataHelper.is_stata_initialized**(_self_)
Wrapper for` pystata.stata.is_initialized()`.
Returns True if Stata is initialized, False otherwise.

### **StataHelper.status**(_self_)
Wrapper for `pystata.stata.status()`.
Prints the status of the Stata instance to the console. Returns None.

### **StataHelper.close_output_file**(_self_)
Wrapper for `pystata.stata.close_output_file()`.
Closes the Stata log file.

### **StataHelper.get_return**(_self_)
Wrapper for `pystata.stata.get_return()`.
Returns the `return` values from the last Stata command as a dictionary.

### **StataHelper.get_ereturn**(_self_)
Wrapper for `pystata.stata.get_ereturn()`.
Returns the `e(return)` values from the last Stata command as a dictionary.


### **StataHelper.get_sreturn**(_self_)
Wrapper for `pystata.stata.get_sreturn()`.
Returns the `sreturn` values from the last Stata command as a dictionary.


### **StataHelper.run**(_self, cmd, **kwargs_)
Wrapper for `pystata.stata.run()`. Runs cmd in the StataHelper window.
<br>**cmd**_(str)_: Stata command to run

### **StataHelper.use(_self, data, columns=None, obs=None, \*\*kwargs_)**
Pythonic method to load a dataset into Stata. equivalent to `use` command in Stata.
<br>**data**_(str)_: The path to the data file to load into Stata.
<br>**columns**_(list or str)_: The columns to load into Stata. If None, all columns are loaded.
<br>**obs**_(int or str)_: The number of observations to load into Stata. If None, all observations are loaded.


### **StataHelper.use_file(_self, path, frame=None, force=False, \*args, \*\*kwargs_)**
Read any pandas-supported file into Stata. Equivalent to `import delimited` in Stata for delimited files. 
This method allows some files that cannot be imported directly into Stata to be loaded.
<br>**path**_(str)_: The path to the file to load into Stata.
<br>**frame**_(str)_: The name of the frame to load the data into. If None, the file name is used.
<br>**force**_(bool)_: Whether to overwrite the existing frame. If False, the frame is appended.

Raises a `ValueError` if the extension is not in the list of supported file types.

Valid file types include:
- CSV
- Excel
- Parquet
- Stata
- Feather
- SAS
- SPSS
- SQL
- HTML
- JSON
- pickle/compressed files
- xml
- clipboard


### **StataHelper.use_as_pandas(_self, frame=None, var=None, obs=None, selectvar=None, valuelabels=None, missinglabels=\_DefaultMissing(), \*args, \*\*kwargs_)**
Read a Stata frame into a pandas DataFrame. Equivalent to `export delimited` in Stata for delimited files.
<br>**frame**_(str)_: The name of the frame to read into a pandas DataFrame. If None, the active frame is used.
<br>**var**_(list or str)_: The variables to read into the DataFrame. If None, all variables are read.
<br>**obs**_(int or str)_: The number of observations to read into the DataFrame. If None, all observations are read.
<br>**selectvar**_(str)_: The variable to use as the index. If None, the index is not set.
<br>**valuelabels**_(bool)_: Whether to use value labels. If True, the value labels are used. If False, the raw values are used.
<br>**missinglabels**_(str)_: The missing value labels to use. If None, the default missing value labels are used.

This method allows some files that stata cannot export directly to be read into a pandas DataFrame. 
In the case of .dta files, this method is significantly faster than using the `pandas.read_stata` method as the dataset 
is first loaded into Stata and then read into a pandas DataFrame, which reduces overhead in directly reading the file in Pandas.


### **StataHelper.save(path, frame=None, var=None, obs=None, selectvar=None, valuelabel=None, missinglabel=None, missval=\_DefaultMissing(), \*args, \*\*kwargs_)**
Save a Stata dataset to a file. Pases the frame to Pandas and saves the file using the Pandas method. Valid file types are
the same as `use_file`.
<br>**path**_(str)_: The path to save the file to.
<br>**frame**_(str)_: The name of the frame to save. If None, the active frame is used.
<br>**var**_(list or str)_: The variables to save. If None, all variables are saved.
<br>**obs**_(int or str)_: The number of observations to save. If None, all observations are saved.
<br>**selectvar**_(str)_: The variable to use as the index. If None, the index is not set.
<br>**valuelabels**_(bool)_: Whether to use value labels. If True, the value labels are used. If False, the raw values are used.
<br>**missinglabels**_(str)_: The missing value labels to use. If None, the default missing value labels are used.
<br>**missval**_(str)_: The missing value labels to use. If None, the default missing value labels are used.

Raises a `ValueError` if the extension is not in the list of supported file types.

### **StataHelper.schedule(_self, cmd, pmap)**
Return the que of commands to be run in parallel (cartesian product). Analogous to the parallel method, but
does not execute the commands.
<br>**cmd**_(str)_: The Stata command to run in parallel.
<br>**pmap**_(dict)_: The parameters to iterate over in the Stata command. Can be any iterable object of any dimension, but
note that the deeper the dimension, the more (potentially redundant) combinations are created.

All keys in pmap must be in cmd, and all placeholders in cmd must be in pmap.

This method creates a queue of string commands to be run in parallel by replacing the bracketed values with their respecitve
values in the cartesian product of the values in pmap.
"Queue" is used loosely here, as the commands are not run sequentially and there is no guarantee of the order in 
which they are run in parallel. 

however, each process's command is labelled with a unique identifier in the order of the queue


### **StataHelper.parallel**(_self, code, values, stata_wildcard=False, max_cores=None, safety_buffer=1)
 **code** _(str)_: The StataHelper code to run in parallel, including placeholders for the values to iterate over.
Placeholders can be either wildcards `*` or brace notation `{}`. If wildcards are used, then the items in `values` must be
ordered in order of the wildcards and its type can be any of list, dict, or tuple. 
If brace notation is used, then `values` must be a dictionary with keys that match the placeholders in the StataHelper code.
<br><br>
e.g. `reg {y} {x} {control}` would require a dictionary with keys `y`, `x`, and `control`.<br>

**values** _(list, dict, tuple)_: The values to iterate over in the StataHelper code. 
If a list or tuple, the order of the values. If a dict, the order only matters if you use wildcards.
In that case, the keys are ignored. Items in sublists are joined with a whitespace `" "` and allow multiple values for a single placeholder.

```python
values = {'x': [['indepvar1', 'indepvar2'], 'indepvar1', 'indepvar2', 'indepvar3']}
stata.parallel("reg y {x}", values)
```
would run the following regressions:
```stata
reg y indepvar1 indepvar2
reg y indepvar1
reg y indepvar2
reg y indepvar3
```
Values can also be conditional statements.

```python
values = {'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'subset': ['var1==1', 'var2==2', 'var3==3']}
stata.parallel("reg y {x} if {subset}", values)
```
would run the following regressions:
```stata
reg y indepvar1 if var1==1
reg y indepvar2 if var2==2
reg y indepvar3 if var3==3
```
Logical operators must be specified in the conditional statement.

```python
values = {'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'subset': ['var1==1 & var2==2', 'var2==2 | var3==3', 'var3==3']}
StataHelper.parallel("reg y {x} if {subset}", values)
```
would run the following regressions:
```stata
reg y indepvar1 if var1==1 & var2==2
reg y indepvar2 if var2==2 | var3==3
reg y indepvar3 if var3==3
``` 
 **max_cores** _(int)_: The maximum number of cores to use. If `None`, then the min of `os.cpus()-safety_buffer` and 
 the total number of combinations is used. If `max_cores` is greater than the number of combinations and the number of
 combinations is greater than the number of cores, then `os.cpus()-safety_buffer` are used.
 
 **safety_buffer** _(int)_: The number of cores to leave open for other processes.
 
 **show_batches** _(bool)_: Prints all the combinations of the values to the console. Useful for debugging. 
 `StataHelper.parallel()` creates a list containing the cartesian product of `values`, and `show_batches` 
 prints this list to the console.
---
## Contributing


## License





---
_Author Collin Zoeller and StataHelper are not affiliated with StataCorp. Stata is a registered trademark of StataCorp 
LLC. While StataHelper is open source, Stata and its Python API Pystata are proprietary software and require a license. 
See [stata.com](https://www.stata.com/) for more information._

###### tags: `Stata` `Python` `Pystata` `Parallelization` `StataHelper` `Documentation`
###### August 2024
