## Extract PertFormer Attention Score 
Relevant codes were documented in "codes/example_attention_compute.ipynb". Briefly, you can input the relevant DNA+ATAC information of a specific gene, and the codes will execute the prediction of Attention Scores around this gene.

* Input Data
  * Example data can be found in "data/info_attention.txt".
  * You need to provide n x 1029-bp DNA sequences and n x 1029-bp ATAC signals, you need to specify the peak that is closest to the TSS, along with the strand of the gene.
  * Deatiled formatting can be found in the .ipynb file.

After you have prepared the input data, run the codes cell by cell in the .ipynb file. The codes will print out the output information, which includes the Attention Score for each ATAC peak.
  * Output data format:
      >chrX_48648355_48648753 <br>
      Attention score:  0.9368568062782288 <br>
      >chrX_48652218_48652494 <br>
      Attention score:  1.2849560976028442 <br>
      ... ...


## PertFormer simulation of perturbations (include Knockout, Knock-in, and etc.)
Relevant codes were documented in "codes/example_gexp_compute.ipynb". Briefly, you can input the relevant DNA+ATAC information of a specific gene, and the codes will execute the prediction of its expression level.

**To perform the *in silico* perturbation, you need to provide the DNA+ATAC profiles before (Ref) and after (KO, KI, and etc.) the perturbation respectively in two files.**

* Input Data format is similar to the previous section, you may refer to the .ipynb file for details.
 
After you have prepared the input data, run the codes cell by cell in the .ipynb file. The codes will print out the gene expressions before and after the perturbation, and you can check with their difference to model the perturbation.
