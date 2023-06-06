"""
a allennlp `Predictor` for generating HANS prediction result 
which combines neutal and contradiction label to a single `non-entailment` label
so that the ouput file can be processed by hans repo's `evaluate_heur_output.py`

plan to use a predictor 
but ipython notebook may be good enough 
by
for i in range(len(dev_set)):
    res = model(**batch)
    ...
"""