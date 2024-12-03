
root=https://raw.githubusercontent.com/BogdanBintu/ChromatinImaging/refs/heads/master/Data/

while IFS= read -r line; do
    wget $root$line
done <data_filenames.txt

mv HCT116_chr21-34-37Mb_6h\ auxin.csv HCT116_chr21-34-37Mb_6h_auxin.csv
mv IMR90_chr21-28-30Mb_cell\ cycle.csv IMR90_chr21-28-30Mb_cell_cycle.csv

