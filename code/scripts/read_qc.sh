#!/bin/bash

set -e

export SINGLEM_METAPACKAGE_PATH=/researchdrive/gkanaan/databases/singlem_db/

num_threads=20
q_score=10
min_length=100

red="\033[0;31m"
nocolor="\033[0m"
bold=$(tput bold)
normal=$(tput sgr0)

print_usage() {
    echo "Usage: read_qc.sh [flags] file1.bam [file2.bam ...]"
    echo "Flags:"
    echo "  -t NUM_THREADS          Number of threads to use"
    echo "  -c CONFIDENCE           Kraken2 confidence threshold to use must be between 0 and 1"
    echo "  -q PHRED SCORE          Minimum phred score a read must pass to be included upstream"
    echo "  -l LENGTH               Minimum length a read must be to be included upstream"
    echo "  -h                      Displays this text"
}

while getopts 't:c:q:l:h' flag; do
    case "${flag}" in
        t) num_threads=${OPTARG};;
        c) conf=${OPTARG};;
        q) q_score=${OPTARG};;
        l) min_length=${OPTARG};;
        *) print_usage
            exit 1;;
    esac
done

shift $((OPTIND - 1))
barcodes=$@

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

# Make the directories we will need
mkdir -p fastqs/raw
mkdir -p fastqs/decontaminated

for file in $barcodes;
do
    (
    if [[ $file != *.bam ]]; then
        echo -e "${red}Error:${nocolor} Skipping ${bold}${file}${normal} as it is not a BAM file. BAM files must end in .bam"
        continue
    fi 
    i=`basename $file .bam`

    echo "Processing {$file}..."

    echo "Making a BAM index..."
    samtools index -@ $num_threads $file

    echo "Converting to FASTQ with methylation data..."
    samtools fastq -@ $num_threads -T ML,MM $file > fastqs/raw/$i.fastq

    echo "Filtering based on length, phred score, and removing human reads..."
    chopper -l $min_length -q $q_score --threads $num_threads --contam /researchdrive/gkanaan/databases/contaminants/contaminants.fasta < fastqs/raw/$i.fastq > fastqs/decontaminated/$i.fastq

    #echo "FASTQ to FASTA then FCS..."
    #mkdir -p fastas/decontaminated/
    #mkdir -p fastas/fcsed/$i
    #seqtk seq -a fastqs/decontaminated/$i.fastq > fastas/decontaminated/$i.fasta
    #./run_fcsadaptor.sh --fasta-input fastas/decontaminated/$i.fasta --output-dir fastas/fcsed/$i --prok --container-engine singularity --image fcs-adaptor.sif

    echo "Profilling reads with SingleM..."
    mkdir -p diversity/raw/$i/
    mkdir -p diversity/decontaminated/$i/
    singlem pipe --reads fastqs/decontaminated/$i.fastq --threads $num_threads --taxonomic-profile-krona diversity/decontaminated/$i/reads-singlem.krona -p diversity/decontaminated/$i/reads-singlem-profile.tsv

    singlem pipe --reads fastqs/raw/$i.fastq --threads $num_threads --taxonomic-profile-krona diversity/raw/$i/reads-singlem.krona -p diversity/raw/$i/reads-singlem-profile.tsv

    echo -e "${bold}DONE PROCESSING ${file}!${nocolor}"
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

echo "Running NanoPlot and NanoComp..."

mkdir -p sequencing_analysis/nanoplot/
mkdir -p sequencing_analysis/nanocomp/

NanoPlot --threads $num_threads -f pdf --N50 -o sequencing_analysis/nanoplot --ubam $barcodes
NanoComp -t $num_threads -o sequencing_analysis/nanocomp -f pdf --ubam $barcodes

echo "Making Krona..."
ktImportKrona -c -o diversity/decontaminated_barcodes_krona.html diversity/decontaminated/*/*.krona
ktImportKrona -c -o diversity/raw_barcodes_krona.html diversity/raw/*/*.krona
