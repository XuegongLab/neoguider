### MixMHCPred and PRIME
### https://doi.org/10.1016/j.cels.2022.12.002
### https://www.sciencedirect.com/science/article/pii/S2405471222004707

# Table S2. List of HLA-I ligands used to train MixMHCpred2.2, related to Figure 2.
wget -c https://ars.els-cdn.com/content/image/1-s2.0-S2405471222004707-mmc4.txt

# Table S4. List of immunogenic and non-immunogenic peptides used to train PRIME2.0, related to Figure 3.
wget -c https://ars.els-cdn.com/content/image/1-s2.0-S2405471222004707-mmc6.xlsx

### MHCmotifAtlas
cp /mnt/d/code/neoguider/database/all_peptides.txt ./mhcmotifatlas.all_peptides.txt.tsv

### NetMHCpan
# download
curl -L -O https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz
# extract (try the simple way first)
tar -xzf NetMHCpan_train.tar.gz
# if tar complains (some systems), use their documented pipeline: uncompress -c NetMHCpan_train.tar.gz | tar xvf -

### MHCflurry 
### paper version: https://data.mendeley.com/datasets/zx3kjzc3yx/3
### software version:
### /home/zhaoxiaofei/.local/share/mhcflurry/4/2.2.0/data_curated/curated_training_data.affinity.csv.bz2
### /home/zhaoxiaofei/.local/share/mhcflurry/4/2.2.0/data_curated/curated_training_data.csv.bz2
### /home/zhaoxiaofei/.local/share/mhcflurry/4/2.2.0/data_curated/curated_training_data.mass_spec.csv.bz2
### /home/zhaoxiaofei/.local/share/mhcflurry/4/2.2.0/data_curated/curated_training_data.no_additional_ms.csv.bz2
mkdir -p  mhcflurry_4_2.2.0/
cp /home/zhaoxiaofei/.local/share/mhcflurry/4/2.2.0/data_curated/curated_training_data.* mhcflurry_4_2.2.0/

for f in $(ls mhcflurry_4_2.2.0/curated_training_data*.csv.bz2); do echo " bzcat $f > ${f/.csv.bz2/.csv} "; done

