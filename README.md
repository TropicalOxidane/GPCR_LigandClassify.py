# GPCR_LigandClassify.py

========================================================

* Go to: https://zhanggroup.org/GLASS/, and download the interactions_active.tsv, ligands.tsv, and targets.tsv files.
* Save the files to a folder called: TSV_2; after having done so, they'll be ready to merge. 

** The following python libraries are required to run the model **

* Python 3.13
* Deepchem 1.x (requires RDKit)
* Scikitlearn (For MLPClassifier)
* Seaborn (For jointplot of XlogP and Molecular Weight)
* TSNE (For visualization of high-dimensional data)

# Autodock simulation after the DNN prediction

** Dopamine D3 Autodock structure example **

![Alt text](D3_example.png)

* Tutorial for Autodock Vina: https://vina.scripps.edu/tutorial/ 
