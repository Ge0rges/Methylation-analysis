set -e 

num_threads=20
mag_dir="/researchdrive/gkanaan/seaice_methylation/mags/"

mkdir mag_eval
coverm genome -m mean -t $num_threads --single fastqs/decontaminated/*.fastq -d $mag_dir -o mag_eval/coverm_mean.tsv
checkm2 predict --threads $num_threads --input $mag_dir --output-directory mag_eval/checkm2_out

export GTDBTK_DATA_PATH=/Accounts/gkanaan/miniconda3/mageval/share/gtdbtk-2.3.2/db
gtdbtk classify_wf --genome_dir $mag_dir --out_dir mag_eval/gtdb --cpus $num_threads --mash_db mag_eval/gtdbtk_mash
rm mag_eval/gtdbtk_mash.msh
rm mag_eval/align/*.gz
