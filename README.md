# SubstitutabilityScore

This package implements the SubstitutabilityScore introduced in the paper "Investigating substitutability of food items in consumption data" by Akkoyunlu et al.

## Requirements 
You need Python 3.7 to run this package. 

## Installation
You can install the package from the repository. 

## Fast execution
The file "XP_substitutions.py" can be executed in order to generate the substitutability scores. The arguments of the main function are:
 ```python
filepath_dico = 'data/dict_cod.p'
filepath_data = 'data/conso_ad.p'
level = 'codsougr' # can take value in ['codgr', 'codsougr', 'codal']
max_meal = 10 # meals larger than max_meal are not taken into account for the computation, lower max_meal speeds up the computation
score = jaccardIndex # can take value in [jaccardIndex, jaccardIndex2]

tyrep = [1,3,5] # can be a list of values of tyrep


```

## Explanation in detail
### Importing the dataset
The input file is a .csv file containing the food consumption entries. First, let us import the dataset. 
```python
import pandas as pd
path_data = '/data/conso.csv'
df = pd.read_csv(path_data)
```
 Let us check the dataframe.
 ```python
df.head(10)
```
The dataframe should have this form. 

<table class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>nomen</th>      <th>jour</th>      <th>tyrep</th>      <th>codsougr</th>      <th>codsougr_name</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>110006</td>      <td>1</td>      <td>1</td>      <td>3499</td>      <td>coffee</td>    </tr>    <tr>      <th>1</th>      <td>110006</td>      <td>1</td>      <td>1</td>      <td>301</td>      <td>sugar</td>    </tr>    <tr>      <th>2</th>      <td>110006</td>      <td>1</td>      <td>1</td>      <td>11</td>      <td>bread</td>    </tr>    <tr>      <th>3</th>      <td>110006</td>      <td>1</td>      <td>1</td>      <td>1599</td>      <td>margarine</td>    </tr>    <tr>      <th>4</th>      <td>110006</td>      <td>1</td>      <td>3</td>      <td>172</td>      <td>beef</td>    </tr>    <tr>      <th>5</th>      <td>110006</td>      <td>1</td>      <td>3</td>      <td>237</td>      <td>beans</td>    </tr>    <tr>      <th>6</th>      <td>110006</td>      <td>1</td>      <td>3</td>      <td>11</td>      <td>bread</td>    </tr>    <tr>      <th>7</th>      <td>110006</td>      <td>1</td>      <td>3</td>      <td>412</td>      <td>dessert</td>    </tr>    <tr>      <th>8</th>      <td>110006</td>      <td>1</td>      <td>3</td>      <td>314</td>      <td>spring water</td>    </tr>    <tr>      <th>9</th>      <td>110006</td>      <td>1</td>      <td>5</td>      <td>211</td>      <td>sea fish</td>    </tr>  </tbody></table>


### Preprocessing
Now the objective is to convert this single item entry dataset to a list of meals. It is possible to select the level of foods to consider i.e ('codgr', 'codsougr', 'codal') if specified in the dataset file. It is also possible to select specific meal types i.e breakfast, lunch or/and dinner. By default, the algorithm considers all meals. 
 ```python
from substitutability import *
level = 'codsougr'

tyrep = [3,5] # lunch and dinner
meals = getMeals(df, level, tyrep)

# If you want to compute the substitutability scores on all meals
allMeals = getMeals(df, level)
```
The first two meals are:
  ```python
allMeals[:1]
# >> [[3499, 301, 11, 15,  1599]
#     [172,237,11,412, 314]]
```

###
