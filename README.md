This is a Variational Inference implementation of the Mention Pair Annotations (MPA) model.
The software requires JAVA and the Apache Commons Math external library.

The code is accompanied by an example input file (example.csv) which assumes the following structure:
mention_id,annotator_id,gold,annotation
ne9399,annotator1,DO(ne9398),DO(ne9398)
ne9399,annotator2,DO(ne9398),DO(ne9395)
ne9399,annotator3,DO(ne9398),DO(ne9398)
ne9399,annotator4,DO(ne9398),DO(ne9396)
...

The header describes the id of the mention, the id of the annotator, the gold (expert) label and the annotation label provided by the annotator. The code will automatically extract the class from the annotation label (e.g.: DO).

Running the code produces posterior point estimates for all the model parameters. The output is set to show the accuracy of the inferred mention pairs against the gold standard. In also includes the accuracy of a majority vote baseline, computed over 10 random rounds of splitting ties.
