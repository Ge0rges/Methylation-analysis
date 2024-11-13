#!/usr/bin/bash

set -e

mkdir -p anvio

# Paramaters
num_threads=50
num_clusters=10
min_contig_length=1000

# Make contigs DB
assembly=assemblies/coassembly/draft/assembly.fasta
anvi-script-reformat-fasta -l $min_contig_length -o anvio/filtered_assembly.fasta $assembly
assembly=anvio/filtered_assembly.fasta

anvi-gen-contigs-database -f $assembly  -T $num_threads -n "Sea-ice Coassembly" -o anvio/CONTIGS.db --full-gene-calling-report anvio/gene-calls.txt

# Populate the contigs DB
anvi-run-hmms -c anvio/CONTIGS.db -T $num_threads
anvi-run-scg-taxonomy -c anvio/CONTIGS.db -T $num_threads


# Barcode to sample matching
declare -A sample_names
sample_names['barcode01']='S2_1'
sample_names['barcode02']='S2_2'
sample_names['barcode03']='S2_3'
sample_names['barcode04']='Control'
sample_names['barcode05']='S3_1'
sample_names['barcode06']='S3_2'
sample_names['barcode07']='S3_3'
sample_names['barcode08']='S4_1'
sample_names['barcode09']='S4_2'
sample_names['barcode10']='S4_3'
sample_names['barcode11']='IC3_1'
sample_names['barcode12']='IC3_2'
sample_names['barcode13']='IC3_3'
sample_names['barcode14']='IC3_4'
sample_names['barcode15']='S4_1_PCR'
sample_names['barcode16']='S4_2_PCR'
sample_names['barcode17']='S4_3_PCR'
sample_names['barcode18']='IC3_1_PCR'
sample_names['barcode19']='IC3_2_PCR'
sample_names['barcode20']='IC3_3_PCR'
sample_names['barcode21']='IC3_4_PCR'
sample_names['control']='Control'

# Make profiles then merge
for fastq in fastqs/decontaminated/*.fastq
do
    name=$(basename $fastq .fastq)
    # Do mapping to reformated fasta
    minimap2 -t $num_threads -ayx lr:hq $assembly $fastq > anvio/"${name}.sam"

    # Make profile
    anvi-init-bam anvio/$name.sam -o anvio/$name.bam -T $num_threads
    anvi-profile -i anvio/$name.bam -c anvio/CONTIGS.db -T $num_threads -S "${sample_names[$name]}"

done

anvi-merge anvio/*/PROFILE.db -o anvio/SAMPLES-MERGED -c anvio/CONTIGS.db -S "Sea_ice_metagenome"

# Keep the BAMs for methylation
mkdir -p bams/aligned/assembly
mv anvio/*.bam bams/aligned/assembly

# Cleanup
rm -r anvio/barcode*
rm anvio/control*
rm $assembly

# Make metabins then summarize
anvi-cluster-contigs -p anvio/SAMPLES-MERGED/PROFILE.db  -c anvio/CONTIGS.db --driver concoct -T $num_threads --clusters $num_clusters -C METABINS --just-do-it

# Summarize
anvi-summarize -p anvio/SAMPLES-MERGED/PROFILE.db -c anvio/CONTIGS.db -C METABINS -o ASSEMBLY_SUMMARY

# Make CONTIGS DB from genomes for annotations
mkdir -p anvio/genomes
for genome in mags/*.fna;
do
    name=`basename $genome .fna`
    if [[ ! $genome == *"-contigs"* ]]; then
        anvi-gen-contigs-database -f $genome -o anvio/genomes/$name-contigs.db -T $num_threads
        anvi-run-ncbi-cogs -c anvio/genomes/$name-contigs.db -T $num_threads
        anvi-run-kegg-kofams -c anvio/genomes/$name-contigs.db -T $num_threads
    fi
done

# Things to run manually for/after refinning:
#anvi-run-kegg-kofams -c anvio/CONTIGS.db -T $num_threads
#anvi-run-ncbi-cogs -c anvio/CONTIGS.db -T $num_threads
#anvi-script-get-collection-info -p anvio/SAMPLES-MERGED/PROFILE.db -c anvio/CONTIGS.db --list-collections
#anvi-refine -p anvio/SAMPLES-MERGED/PROFILE.db -c anvio/CONTIGS.db -C METABINS -b Bin_name
#anvi-summarize -p anvio/SAMPLES-MERGED/PROFILE.db -c anvio/CONTIGS.db -C METABINS -o REFINED_SUMMARY
