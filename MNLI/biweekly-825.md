# Bi-weekly Report, 2020-08-25
- online version: [Bi-weekly Report, 2020-08-25](/8WR1rWccSii8vF731JbhzA)
- Yan-Tong lin

---

## Overview
<!-- .slide: style="font-size: 30px;" -->
- Last Two Weeks
- Milestones
- KPI table
- Lesson Learn

---

## Last Two Weeks (8/11-8/25)
<!-- .slide: style="font-size: 30px;" -->
- [Paper List, 2020 Summer](/1btwWqvPSI6-Jd3JFOk_rw)
- [Graph Embedding Net and Graph Matching Net]()
- [Modularized Model with AllenNLP](/cQb4Z6yTS3eVQplFFbS3NA)
    - Coding
- [New Test Set]()
    - TODO

---

## Graph Embedding Net (like InferSent, Training)
<!-- .slide: style="font-size: 10px;" -->
![](https://i.imgur.com/ZETKI4b.png =60%x)
- `Graph Matching Networks for Learning the Similarity of Graph Structured Objects, DeepMind 2019`

---

## Graph Matching Net (Under construction)
<!-- .slide: style="font-size: 10px;" -->
![](https://i.imgur.com/gMjUjqj.png =50%x)
- `Graph Matching Networks for Learning the Similarity of Graph Structured Objects, DeepMind 2019`



---

## Modularized Model with AllenNLP
<!-- .slide: style="font-size: 30px;" -->
- [Modularized Model with AllenNLP](/cQb4Z6yTS3eVQplFFbS3NA)

---

## Current Training Model (Graph Embedding Net)
<!-- .slide: style="font-size: 30px;" -->
- BERT embedding ("Not" Trainable)
- Relational Graph Convolution Network
- "No" Cross Attention
- Global Attentive Pooling
- Train: 0.618
- Dev:   0.652

---

## Current Training Model Performance
<!-- .slide: style="font-size: 30px;" -->

| model\acc | Train  | Matched | MisMatched |
| -------------- | ------ | ------- | ---------- |
| random/majority     | ~0.33 | ~0.33  |~0.33     |
| aESIM          | -      | 73.9    | 73.9       |
| BERT + SingleTaskAdaptor        | -      | 85.4    | 85.0       |
| RoBERTa        | -      | 90.8    | 90.2       |
| Glove-GAT3-CrossAtt-MaxPool     | 0.6323 | 0.6116  | 0.6252     |
| BERT-RGCN3-AttPool-VecComp (6th epoch)     | 0.618 | 0.652  | -     |


----

## Training Curve (On MNLI)
<!-- .slide: style="font-size: 25px;" -->
- weird, dev >> train
    - blue = validation
    - orange = train
- may because MNLI trainset size >> devset size
![](https://i.imgur.com/6nmvuna.png)

----

## Parameter Distribution
<!-- .slide: style="font-size: 25px;" -->
- gate_nn discussion
    - smaller bias
    - concentrated higher weights
![](https://i.imgur.com/rjDjx1K.png)


----

## Error Analysis
<!-- .slide: style="font-size: 25px;" -->
- after training complete
- bad training performance
    - no cross attention (fixed in `Graph Matching Net`)
    - BERT cannot improve much (try `Elmo` or `Glove` to see how performance change)

---

## New Test Set
<!-- .slide: style="font-size: 25px;" -->
- Template NLI
:::success
|   | Premise | Hypothesis |
| -------- | -------- | -------- |
| Template     | S + NoNeg + VP    | S + Neg + VP    |
| Generated Instance     | Allen does like to eat pizza     |  Allen does not like to eat pizza    |
:::
- Rule based extention 
    - like Amenda's 


---

## KPI table
<!-- .slide: style="font-size: 30px;" -->
- Now the framework is finished any model can be added and experimented with ease
- But need more GPU for experiments to be done quickly in parallel

 
| Item | ETC | PV  | EV  | SPI |
| ---- | --- | --- | --- | --- |
| ~~Framework~~                                            | **8/15**     | 100 | 100 | 100 |
| ~~BERT+Graph Embedding Net(RGCN)+Global Attention Pool~~ | **8/20**      | 100 | 100 | 100
| **BERT+Graph Matching Net(RGCN)+Global Attention Pool**      | **8/20**     | 100 | 50 | 50
| Experiments      | 8/25      | 100 | 10 | 10


---

## Milestones 
<!-- .slide: style="font-size: 30px;" -->
| Item                                                     | ETC |
| -------------------------------------------------------- | --------- |
| Experiments (GMN, improvements)                                             | 9/15 |
| Composition                                             | 9/19 |
| EACL due                                             | 9/20 |

Note: 
Dates
- -7/20 Survey(-7/15) / Model Design
- -8/15 Implementation and Experiments
- -8/20 Compose the Paper
- Submission Due on 
    - TAAI, Due 8/31
    - AAAI, Due 9/5
    - EACL, Due 9/20

---

## Lesson Learn during Internship
- Knowledge about NLP tasks
- State of the Art NLP Methods
- Modularization + Dependency Injection Coding Style
- Implementation of Graph Neural Networks

---

## Thanks

---

## Reference
<!-- .slide: style="font-size: 20px;" -->
- Li, Yujia, et al. "Graph matching networks for learning the similarity of graph structured objects." arXiv preprint arXiv:1904.12787 (2019).
- Conneau, Alexis, et al. "Supervised learning of universal sentence representations from natural language inference data." arXiv preprint arXiv:1705.02364 (2017).
- Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European Semantic Web Conference. Springer, Cham, 2018.
- Veličković, Petar, et al. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- Gardner, Matt, et al. "Allennlp: A deep semantic natural language processing platform." arXiv preprint arXiv:1803.07640 (2018).

---