# From https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 (Rao et al. 2014) 


##################
# Download data on a login node, as compute nodes don't have internet access. 

##################
# GM12878. I used the file that includes contacts from both the primary replicate experiments, in which contacts were filtered if FASTQ<30. 
# Others are kept here for convenience if desired at a later date. 

# Raw, no FASTQ cutoff, primary
# curl https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary.hic -o ./GM12878_insitu_primary.hic

# FASTQ>=30, primary
#curl https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary.hic -o ./GM12878_insitu_primary_30.hic

# Raw, no FASTQ cutoff, primary + replicate combined 
# curl https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined.hic ./GM12878_insitu_primary+replicate_combined.hic


# FASTQ>=30, primary + replicate combined
#curl https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30.hic -o ./GM12878_insitu_primary+replicate_combined_30.hic

curl https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FIMR90%5Fcombined%5F30.hic -o ./IMR90_insitu_combined_30.hic

##################
# Convert .hic file to a .mcool file and get the 20kb resolution contact data using a compute node, as it takes a long time & is computationally intensive => would otherwise disrupt the login node. 
sbatch --wait hic2cool_IMR90_hg19.sh

#rm GM12878_insitu_primary+replicate_combined_30.hic 
#rm IMR90_insitu_combined_30.hic

