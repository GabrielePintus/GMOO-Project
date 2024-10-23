# Setup file

The setup file is a python script that defines all the things needed for the script to work properly. 
The default setup file is `setup-default.py`. You can create your own setup file by extending the default setup file and chaning what you need.
Example
```python
from setup-default import *

# Set custom domain for the chromosome
Chromosome.set_domain({
    'hidden_size' : (10, 100),
})
```