# Syn NLI Architecture

## Preprocess 
- raw data
    - {sent1, sent2, gold label}
- parse with Stanza 
    - {parsed, }
- saving to files
    - save as dictionary
    - {"xh", "xp", "l", "deph", "depp", "lh", "lp"}

## Model
- data 
    - p: (batch, pad id_list)
    - h: (batch, pad id_list)
    - label: (batch, id)
- contextualized encoder 
    -
- 


```=python

import preprocess
import 
```