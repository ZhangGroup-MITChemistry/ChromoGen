# hg19 aligned data bigWig data. ENCODE database accession ENCFF901GZH. 
curl -o ../../data/outside/GM12878.bigWig https://encode-public.s3.amazonaws.com/2017/09/06/e2259e48-add5-4b57-bc75-e27b290b954f/ENCFF901GZH.bigWig

# hg19 genome from UCSC genome browser
curl -o ../../data/outside/hg19.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip ../../data/outside/hg19.fa.gz 

