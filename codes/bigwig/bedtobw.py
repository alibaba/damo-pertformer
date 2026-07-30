import os

bed_files = os.listdir('bed/')
print(len(bed_files))

cell = 0
for bed_name in bed_files:
    cell+=1
    barcode = bed_name.split('.')[0]
    print('Converting ' + bed_name + ' to bigWig ...', cell)
    
    os.system('bedtools bedtobam -g raw/hg38_size.bed -i bed/' + bed_name + ' > bam/' + barcode + '.bam')
    os.system('samtools index bam/' + barcode + '.bam')
    os.system('bamCoverage --bam bam/' + barcode + '.bam -o bw/' + barcode + '.bw -p 32 --binSize 10 --normalizeUsing CPM')
        
    print('\n')