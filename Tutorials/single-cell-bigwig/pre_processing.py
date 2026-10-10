import os
import scanpy as sc
import csv

scRNA = sc.read_10x_h5('raw/input_multiome.h5', genome=None, gex_only=False)
scRNA.var_names_make_unique
scRNA.var["feature_types"].unique()
is_atac = scRNA.var["feature_types"] == "Peaks"
scRNA[:, is_atac].write_h5ad("raw/atac.h5ad")
scRNA[:, ~is_atac].write_h5ad("raw/rna.h5ad")

scATAC = sc.read_h5ad("raw/atac.h5ad")
print(scATAC.shape)

barcode_dict = {}
for i in range (scATAC.shape[0]):
    barcode  = scATAC[i].obs_names[0]
    barcode_dict[barcode] = open('bed/' + barcode + '.bed', 'a')
    
chromatin_file = open('raw/hg38_size.bed', 'r', encoding='utf-8')
chromatin_size = [0]
for row in chromatin_file:
    chromatin_size = chromatin_size + [row.split('\t')[0]]
chromatin_size = chromatin_size[1:]
print(chromatin_size, len(chromatin_size))

f_tsv = open('raw/human_brain_3k_atac_fragments.tsv', 'r', encoding='utf-8')

count = 0
for row in f_tsv:
    
    info    = row.split('\t')
    chrom   = info[0]
    barcode = info[3]
    if barcode in barcode_dict and chrom in chromatin_size:
        barcode_dict[barcode].write(info[0]+'\t'+info[1]+'\t'+info[2]+'\t'+info[4])
        barcode_dict[barcode].flush()
    count+=1
    if count%1000000==0:
        print(count)
        
bed_files = os.listdir('bed/')
print(len(bed_files))