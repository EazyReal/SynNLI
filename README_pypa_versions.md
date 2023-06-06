# Version Log

---

## 8/27 todo work
- finish Q-test construciton + parsing
- Predition on Q-Test
- Gate Visualizer
- Attention Visualizer
- BiMPM->Traditional GMN version
    - test different matching cut off
- no dropout version
    - In alber, dropout hurs

## 8/27

### Pressing Issues
- ANLI transformer training loss nan
- HGT code migrationi fail (for old version message passing)
- Test Set and Performance
- Paper Composition
- Fix Half Trainning
- Use TWCC
- add other matching module (minus)

### DONE
- TF Board Add Entropy
- MNLI/ANLI GMN

### TODO
- Test Set (clean template)
    - proof bert cannot do not well
        - trivial, since it is pretrained on big data set for word sense 
    - proof tramsformer encoder cannot (train from scratch
        - may not actually be the case
    - proof RGCN + cross attentioin GMN can
- Predictor, Internal Attention Visualization
    - rte_predictor.py
- More Modules
    - RGCN without + self
    - HGT
    - AttentiveSumDiff
    - vector_diff.py for tensors
- Fix cannot use Half
- repo
    - requirements.txt, setup.py
- docker and environment setup
- TODO files:
    - requirements.txt
    - setup.py
    - attetivesumdiff.py 
    - vector_comp.py
    - predictor.py
    - demo code
    - attention visualization
    - half precision with sparse tensor...
- use computational resuorces
- for better GNN
    - https://ogb.stanford.edu/docs/leader_nodeprop/

---