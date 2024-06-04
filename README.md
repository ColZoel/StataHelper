
# Stata
## A Simplified Stata wrapper and Parallelization Library for Python

### Table of Contents
- [Installation](#installation)
- [Introduction](#introduction)
- [Parallelization](#parallelization)
    - [Method 1: Wildcards](#method-1-wildcards)
    - [Method 2: Brace Notation](#method-2-brace-notation)
        - [Stata Wildcards](#stata-wildcards)
    - [Multi-level Fixed Effects](#multi-level-fixed-effects)
    - [Multiline Stata Code](#multiline-stata-code)
    - [Conditional Statements](#conditional-statements)
    - [Unified Interface](#unified-interface)
- [Documentation](#documentation)
- [Class: Stata](#class-stata)
    - [Stata.run](#statarunself-code)
    - [Stata.parallel](#stataparallelself-code-values-stata_wildcardfalse-max_coresnone-safety_buffer1-show_batchesfalse)
    
## Installation
There are two ways to install the package. The first is to install the package from PyPi using pip. 
The second is to clone the repository.

### From PyPi
```bash
pip install Stata
```

```bash 
pip install git+
git clone 
cd Stata
pip install .
```
### Dependencies
- Python 3.9+
- Stata 16+
- Pandas
- Numpy
- yaml



## Introduction
Stata is a Python package that simplifies the process of running Stata code from Python.
The package provides a simple interface to run Stata code, pass data between Python and Stata, 
and parallelize Stata code. The package is designed to be simple to use and easy to integrate into existing Python codebases.

Let's consider a use case.

Suppose you have a set of regressions you want to run in which you change the dependent variable,
independent variables, or control variables. In Stata this would require several foreach loops over the variables to 
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
result holds across different groups of the population, fixed-effect levels, or time periods. Stata is a powerful tool
for this type of analysis, but is only designed to run a single regression at a time. 

For the sake of argument, let's say that Stata takes X seconds to run a single regression.
If we have 3 dependent variables, 3 independent variables, and 3 control variables, we would need to run 27 regressions.
This would take 27X seconds to run. Let's say we want to see if the result holds for two segments of the population
(e.g. heterogeneous effects) 
, so now we have 3 dependent variables, 3 independent variables, 3 control variables, and 2 segments = 54 regressions, 
requires another foreach loop. This would take 54X seconds to run. As the number of variations increases, the time to run the regressions increases
exponentially, each forloop has time complexity O(n), so the total time complexity is O(n^4).

This inefficiency is where Stata comes in. Stata allows you to parallelize the regression groups,
reducing the time to run the regressions.

```python
from StataHelper import Stata

stata = Stata(splash=False)
results = stata.parallel("reg * * *", [['depvar1', 'depvar2', 'depvar3'],
                                       ['indepvar1', 'indepvar2', 'indepvar3'],
                                       ['controlvar1', 'controlvar2', 'controlvar3']])

```
The idea of parallelization is that we divide the number of regressions into smaller ques and run them simultaneously across
multiple cores. This reduces the time to run the regressions. If you have those 27 regressions and divide them evenly 
across 3 cores, you would reduce the time to run the regressions by 3X.

Additionally, Stata provides users a simplified interface to interact with pystata, can read and write data that 
cannot be imported directly to Stata, like Apache Parquet files, and can run Stata code from a string or a file.


## Parallelization
StataHelper provides a simple interface to parallelize Stata code. Just as with pystata's `run` method, 
you may pass a string of Stata code to the `parallel` method. Stata is designed to read placeholders in the stata
code for the values you wish to iterate over. There are two methods to do this:

### Method 1: Wildcards
Complex stata code can be simplified quickly with wildcards. Consider our previous example.

```python
import StataHelper

stata = StataHelper.Stata(splash=False)
values = [['depvar1', 'depvar2', 'depvar3'],
          ['indepvar1', 'indepvar2', 'indepvar3'],
          ['controlvar1', 'controlvar2', 'controlvar3']]

results = stata.parallel("reg * * *", values)
```
In this example, the `*` wildcard is used to iterate over the dependent variables,
independent variables, and control variables in that order. 
In other words, the first wildcard corresponds to values[0], the second wildcard corresponds to values[1],
and the third wildcard corresponds to values[2]. This is repeated for any number of wildcards in the Stata code.

The iterable does not have to be a list with this notation, you can also input a dictionary (in which case the keys are 
ignored and order still matters) or a tuple. 



### Method 2: Brace Notation
The previous example can be rewritten using bracket notation.

```python
from StataHelper import Stata

stata = Stata(splash=False)
values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'control': ['controlvar1', 'controlvar2', 'controlvar3']}

results = stata.parallel("reg {y} {x} {control}", values)
```
This notation may be moore intuitive for some user. Note that this notation requires a dictionary with keys that match the
placeholders in the Stata code. This notation has the advantage that the arguments are position 
agnostic, so the order of the keys in the dictionary does not matter. This means that
the following dictionary would produce the same results as the previous example.

```python
values = {'control': ['controlvar1', 'controlvar2', 'controlvar3'],
          'x':['indepvar1', 'indepvar2', 'indepvar3'], 
          'y': ['depvar1', 'depvar2', 'depvar3']}
```

#### Stata Wildcards
If you use the native Stata wildcard `*`, you can set Stata to ignore the wildcard by setting `stata_wildcard=True`. 
For example if you wanted to run

```Stata
reg y ind* cont*
```
which runs the regression with the dependent variable `y`, all independent variables `indvar1`, `indvar2`, `indvar3`,
and all control variables `contvar1`, `contvar2`, `contvar3`, you would set `stata_wildcard=True`.

```python
from StataHelper import Stata as Stata

stata = Stata(splash=False)
values = {'y': ['depvar1', 'depvar2', 'depvar3']}

results = stata.parallel("reg {y} ind* contr*", values, stata_wildcards=True)
```

### Multi-level Fixed Effects
Let's say you wanted to run a series of regressions but vary the level of fixed effects. You would approach this by 
simply introducing a sublist into the fixed effects list. In the following example, we'll use the `reghdfe` command to
run a series of regressions with varying high-dimensional fixed effects.

```python
from StataHelper import Stata

values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'fixed_effects': [['fe1', 'fe2'], ['fe1'], ['fe1', 'fe5']]}
stata = Stata(splash=False)
stata.run("ssc install reghdfe")
results = stata.parallel("reghdfe {y} {x} absorb({fixed_effects})", values)
```

### Multiline Stata Code
You can also pass multiline Stata code to the `parallel` method, just as you can with `pystata.stata.run`.

```python
import StataHelper

stata = StataHelper.Stata(splash=False)
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

stata = StataHelper.Stata(splash=False)
values = {'y': ['depvar1', 'depvar2', 'depvar3'],
          'x': ['indepvar1', 'indepvar2', 'indepvar3'],
          'control': ['controlvar1', 'controlvar2', 'controlvar3'],
          'subsets': ['var4<=2023 & var5==1', 'var4>2023 | var5==0']}
results = stata.parallel("reg {y} {x} {control} if {subsets}", values)
```

## Unified Interface
You can interact with Stata in nearly the same way you would interact with pystata. In pystata you would configure the
pystata instance as follows:
    
```python
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
stata.run("di hello world")
stata.run("use data.dta")
stata.run("reg y x")
config.close_output_file()  # Close the Stata log
```

Notice how we have to configure the stata instance before we can even call import the `stata` module, 
and the stata instance requires a separate `config` object to be configured. 

In StataHelper, you can configure the Stata instance directly in the constructor. 

```python
from StataHelper import Stata
stata = Stata(splash=False,
              edition='mp',
              set_graph_format='svg',
              set_graph_size=(800, 600),
              set_graph_show=False,
              set_command_show=False,
              set_autocompletion=False,
              set_streaming_output=False,
              set_output_file='output.log')
stata.run("di hello world")
```
Alternatively, you can pass in a YAML file with the configuration settings. 

```yaml
# config.yaml
path: "C:/Program Files/Stata17/StataMP-64.exe"
splash: False
edition: mp
set_graph_format: svg
set_graph_size: [800, 600]
set_graph_show: False
set_command_show: False
set_autocompletion: False
set_streaming_output: False
...
```

```python
import StataHelper
stata = StataHelper.Stata(config='config.yaml')
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
stata = StataHelper.Stata(config=dict)
```
All values not specified either as an argument or in the YAML file default to the pystata defaults. See the
[pystata documentation](https://www.stata.com/python/pystata18/config.html).

# Documentation
> ## Class: Stata
> **Stata**(_self, config=None,
                 edition=None,
                 splash=None,
                 set_graph_format=None,
                 set_graph_size=None,
                 set_graph_show=None,
                 set_command_show=None,
                 set_autocompletion=None,
                 set_streaming_output=None,
                 set_output_file=None)_
>
>> **Config**: A dictionary or YAML file with the configuration settings for the Stata instance. Supercedes any inputted 
arguments.
>>
>>**edition**_(str)_ : The edition of Stata to use. This depends on the license you have. If you have multiple editions installed,
you can specify which edition to use.
>>
>> **splash**_(bool)_:  Whether to show the splash screen when Stata is opened. It is recommended not use this when running parallel, 
as it will repeat for every core that is opened.
>> 
>> **set_graph_format**: <br>pystata.config.set_graph_format. The format of the graphs to be saved.
>>
>> **set_graph_size**: <br>pystata.config.set_graph_size. The size of the graphs to be saved.
>>
>> **set_graph_show**: <br>pystata.config.set_graph_show. Whether to show the graphs in the Stata window.
>>
>> **set_command_show**: <br>pystata.config.set_command_show. Whether to show the commands in the Stata window.
>>
>> **set_autocompletion**: <br>pystata.config.set_autocompletion. Whether to use autocompletion in the Stata window.
>>
>> **set_streaming_output**: <br>pystata.config.set_streaming_output. Whether to stream the output to the console.
>>
>> **set_output_file**: <br>pystata.config.set_output_file. Where to save the Stata log file.
>
>>### **Stata.run**(_self, code_)
>> **code**: pystata.stata.run(). Runs the Stata code in the Stata window.
>
>> ### **Stata.parallel**(_self, code, values, stata_wildcard=False, max_cores=None, safety_buffer=1, show_batches=False_)
>> **code** _(str)_: The Stata code to run in parallel, including placeholders for the values to iterate over.
Placeholders can be either wildcards `*` or brace notation `{}`. If wildcards are used, then the items in `values` must be
ordered in order of the wildcards and its type can be any of list, dict, or tuple. 
If brace notation is used, then `values` must be a dictionary with keys that match the placeholders in the Stata code.
<br><br>
e.g. `reg {y} {x} {control}` would require a dictionary with keys `y`, `x`, and `control`.<br>
e.g. `reg * * *` would require a list, dict, or tuple with the dependent variables, independent variables, and control variables in that order.
<br><br>
> **values** _(list, dict, tuple)_: The values to iterate over in the Stata code. 
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
stata.parallel("reg y {x} if {subset}", values)
```
would run the following regressions:
```stata
reg y indepvar1 if var1==1 & var2==2
reg y indepvar2 if var2==2 | var3==3
reg y indepvar3 if var3==3
```
> **stata_wildcard** _(bool)_: Whether to use the Stata wildcard `*` in the Stata code. If set to `True`, the Stata wildcard
> 
> **max_cores** _(int)_: The maximum number of cores to use. If `None`, then the min of `os.cpus()-safety_buffer` and 
> the total number of combinations is used. If `max_cores` is greater than the number of combinations and the number of
> combinations is greater than the number of cores, then `os.cpus()-safety_buffer` are used.
> 
> **safety_buffer** _(int)_: The number of cores to leave open for other processes.
> 
> **show_batches** _(bool)_: Prints all the combinations of the values to the console. Useful for debugging. 
> `Stata.parallel()` creates a list containing the cartesian product of `values`, and `show_batches` 
> prints this list to the console.