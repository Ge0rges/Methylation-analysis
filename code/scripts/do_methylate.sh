#!/bin/bash

set -e

source /localdata/local/miniconda3/etc/profile.d/conda.sh

num_threads=20
coverage=5
agg=false

red="\033[0;31m"
nocolor="\033[0m"
bold=$(tput bold)
normal=$(tput sgr0)

print_usage() {
    echo "Usage: analyze_methylation.sh [flags] -f <FASTA_GENOMES_DIRECTORY> <FASTQ_METHYLATED_READS>"
    echo "Flags:"
    echo "  -c COVERAGE             Coverage filter to apply in modkit dmr"
    echo "  -t NUM_THREADS          Number of threads to use"
    echo "  -a AGGREGAGATE          If -a is specified, aggregate top/middle/bottom BAMs"
    echo "  -h                      Displays this text"
}

while getopts 't:c:f:ha' flag; do
    case "${flag}" in
        c) coverage=${OPTARG};;
        t) num_threads=${OPTARG};;
        f) genome_dir=${OPTARG};;
        a) agg=true;;
        *) print_usage
            exit 1;;
    esac
done


shift $((OPTIND - 1))
methylated_reads=$@
if $agg; then
    methylation_dir="methylation_${coverage}_agg"
else
    methylation_dir="methylation_${coverage}"
fi

if [ $# -eq 0 ]; then
    echo -e "${red}ERROR:${nocolor} no file provided"
    print_usage
    exit 1
fi

N=1

if [ $num_threads -gt 20 ]; then
    N=$((N-N%20))
    N=$(($num_threads/20))
    num_threads=20
fi

# If $genome_dir is a directory
if [ -d "$genome_dir" ]; then
    genome_dir="$genome_dir/*.fna"
fi 

for genome in $genome_dir;
do

    (
    if [[ $genome != *.fna ]]; then
        echo -e "${red}Warning:${nocolor} Skipping genome ${bold}${genome}${normal} as it is not a FNA file. FNA files must end in .fna"
        continue
    fi

    genome_name=`basename $genome .fna`

    mkdir -p $methylation_dir/$genome_name
    for reads in $methylated_reads;
    do
        if [[ $reads != *.fastq ]]; then
            echo -e "${red}Warning:${nocolor} Skipping reads ${bold}${reads}${normal} as it is not a FASTQ file. FASTQ files must end in .fastq"
            continue
        fi

        reads_name=`basename $reads .fastq`

        # Map
        if [ ! -f "$methylation_dir/$genome_name/$reads_name.bam" ]; then
            echo "Mapping methylated reads onto MAG for " $reads_name
            minimap2 -t $num_threads -ayx lr:hq $genome $reads > $methylation_dir/$genome_name/$reads_name.sam
            samtools view -bS -@ $num_threads $methylation_dir/$genome_name/$reads_name.sam | samtools sort -@ $num_threads -o $methylation_dir/$genome_name/$reads_name.bam
            samtools index -@ $num_threads $methylation_dir/$genome_name/$reads_name.bam
            rm $methylation_dir/$genome_name/$reads_name.sam
        fi

    done


    if [ ! -f "$methylation_dir/$genome_name/top.bam" ] && $agg; then
        # Merge samples together
        samtools merge -@ $num_threads -o "$methylation_dir/$genome_name/top.bam"  $methylation_dir/$genome_name/barcode{01,05,08}.bam
        samtools merge -@ $num_threads -o "$methylation_dir/$genome_name/middle.bam" $methylation_dir/$genome_name/barcode{02,06,09}.bam
        samtools merge -@ $num_threads -o "$methylation_dir/$genome_name/bottom.bam" $methylation_dir/$genome_name/barcode{03,07,10}.bam

        samtools index -@ $num_threads "$methylation_dir/$genome_name/top.bam"
        samtools index -@ $num_threads "$methylation_dir/$genome_name/middle.bam"
        samtools index -@ $num_threads "$methylation_dir/$genome_name/bottom.bam"

        rm $methylation_dir/$genome_name/barcode{01,02,03,05,06,07,08,09,10}*
    fi

    for bam in $methylation_dir/$genome_name/*.bam
    do
        bam_name=`basename $bam .bam`

        # Pileup
        if [ ! -f "$methylation_dir/$genome_name/$bam_name.bed" ]; then
            echo "Running modkit pileup and summary for " $bam_name
            modkit pileup -p 0.1 $bam $methylation_dir/$genome_name/$bam_name.bed -t $num_threads
            awk -v OFS="\t" '{print $1, $2, $3, $11, $10}' $methylation_dir/$genome_name/$bam_name.bed > $methylation_dir/$genome_name/$bam_name-bedgraph.bed
            modkit summary -t $num_threads -p 0.1 --tsv $bam > $methylation_dir/$genome_name/summary_$bam_name.tsv

            bgzip -k --threads $num_threads $methylation_dir/$genome_name/$bam_name.bed
            tabix $methylation_dir/$genome_name/$bam_name.bed.gz

        fi
    done

    for bam in $methylation_dir/$genome_name/*.bam
    do
        if [ ! -f "$methylation_dir/$genome_name/$bam_name-motifs.tsv" ]; then

            echo "Getting enriched motifs " $genome_name " " $bam_name
            modkit find-motifs --min-coverage $coverage  -i $methylation_dir/$genome_name/$bam_name.bed -r $genome -o $methylation_dir/$genome_name/$bam_name-motifs.tsv --threads $num_threads || echo "Find-motifs failed, probably due to coverage."
        fi

        #echo "Calculating entropy..."
        #modkit entropy --min-coverage $coverage --in-bam $bam -o $methylation_dir/$genome_name/$bam_name-entropyA.tsv --ref $genome --threads $num_threads --base A --base C
    done

    if [ ! -f "$methylation_dir/$genome_name/gene-calls.txt" ]; then
        echo "Getting annotations of genome for " $genome_name
        conda deactivate && conda activate anvio-dev

        contigs_db="anvio/CONTIGS.db"

        #python prodigal_caller.py $genome_dir/$genome_name.fna $methylation_dir/$genome_name/gene-calls.txt $num_threads

        anvi-export-gene-calls -c $contigs_db -o $methylation_dir/$genome_name/gene-calls.txt --skip-sequence-reporting --gene-caller prodigal
        anvi-export-functions -c $contigs_db -o $methylation_dir/$genome_name/function-calls.txt

        conda deactivate && conda activate methylate

    fi

    if $agg; then

        echo "Getting DMRs for " $genome_name
        if [ ! -f $methylation_dir/$genome_name/dmr_by_gene/top_bottom.bed ]; then 

            if $agg; then
                sample_list=(top middle bottom barcode11 barcode12 barcode13 barcode14)
                bed_list=(top moddle bottom barcode11 barcode12 barcode13 barcode14)
                length=${bed_list[@]}

            else
                sample_list=(top middle bottom top middle bottom top middle bottom barcode11 barcode12 barcode13 barcode14)
                bed_list=(barcode01 barcode02 barcode03 barcode05 barcode06 barcode07 barcode08 barcode09 barcode10 barcode11 barcode12 barcode13 barcode14)
                length=${#bed_list[@]}
            fi

            [ ${#bed_list[@]} -ne ${#sample_list[@]} ] && echo "Error: bed_list and sample_list are not the same length." && exit 1 || true 

            for ((i=0; i<${#sample_list[@]}; i++)); do
                sample=${sample_list[$i]}
                bed_name=${bed_list[$i]}
                samples+="-s $methylation_dir/$genome_name/$bed_name.bed.gz $sample "
            done

            tail -n +2 "$methylation_dir/$genome_name/gene-calls.txt" | awk 'BEGIN { FS="\t" } { if ($5 == "f") $5 = "+"; else if ($5 == "r") $5 = "-"; print $2"\t"$3"\t"$4"\t"0"\t"0"\t"$5 }' > $methylation_dir/$genome_name/gene-coordinates.txt

            mkdir -p $methylation_dir/$genome_name/dmr_by_gene
            mkdir -p $methylation_dir/$genome_name/dmr_by_position/

            modkit_args=" -t $num_threads --ref $genome --base C --base A --min-valid-coverage $coverage"
            echo  modkit dmr multi $samples -r $methylation_dir/$genome_name/gene-coordinates.txt -o $methylation_dir/$genome_name/dmr_by_gene/ $modkit_args
            modkit dmr multi $samples -r $methylation_dir/$genome_name/gene-coordinates.txt -o $methylation_dir/$genome_name/dmr_by_gene/ $modkit_args


            for ((i=0; i<$length; i++)); do
                for ((j=i+1; j<$length; j++)); do
                    sample1=${bed_list[i]}
                    sample2=${bed_list[j]}

                    echo modkit dmr pair -a $methylation_dir/$genome_name/$sample1.bed.gz -b $methylation_dir/$genome_name/$sample2.bed.gz -o $methylation_dir/$genome_name/dmr_by_position/$sample1-$sample2 $modkit_args
                    modkit dmr pair -a $methylation_dir/$genome_name/$sample1.bed.gz -b $methylation_dir/$genome_name/$sample2.bed.gz -o $methylation_dir/$genome_name/dmr_by_position/$sample1-$sample2 $modkit_args || echo $sample1-$sample2-$genome_name "failed"
                done
            done
        fi
    fi

    echo -e "${bold}DONE PROCESSING ${genome}!${nocolor}"
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

echo "Cleaning up..."
rm $methylation_dir/*/*gz* $methylation_dir/*/gene-coordinates.txt #$methylation_dir/*/*.bam* 

echo "All done."

