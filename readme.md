# Mention Pair Annotations (MPA) model

This is a variational inference implementation of the *Mention Pair Annotations* (MPA) model presented in

Silviu Paun, Jon Chamberlain, Udo Kruschwitz, Juntao Yu, Massimo Poesio (2018). **A Probabilistic Annotation Model for Crowdsourcing Coreference**. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 

The code is written in JAVA and requires the Apache Commons Mathematics library. The input file (e.g., example.csv) assumes the following structure:

```
mention_id,annotator_id,gold,annotation
ne9399,annotator1,DO(ne9398),DO(ne9398)
ne9399,annotator2,DO(ne9398),DO(ne9395)
ne9399,annotator3,DO(ne9398),DO(ne9398)
ne9399,annotator4,DO(ne9398),DO(ne9396)
...
```

The header describes the id of the mention, the id of the annotator, the gold (expert) label and the annotation label provided by the annotator. The code will automatically extract the class from the annotation label (e.g.: DO).

Running the code produces posterior point estimates for all the model parameters. The output is set to show the accuracy of the inferred mention pairs against the gold standard. It also includes the accuracy of a majority vote baseline, computed over 10 random rounds of splitting ties.

## Reference
```
@InProceedings{paun-EtAl:2018:EMNLP,
  author    = {Paun, Silviu  and  Chamberlain, Jon  and  Kruschwitz, Udo  and  Yu, Juntao  and  Poesio, Massimo},
  title     = {A Probabilistic Annotation Model for Crowdsourcing Coreference},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  month     = {October-November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {1926--1937},
  url       = {http://www.aclweb.org/anthology/D18-1218}
}
```
