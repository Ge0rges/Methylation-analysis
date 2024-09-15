#!/bin/bash

set -e

export SINGLEM_METAPACKAGE_PATH=/researchdrive/gkanaan/databases/singlem_db/

num_threads=1

red="\033[0;31m"
nocolor="\033[0m"
bold=$(tput bold)
normal=$(tput sgr0)
stop_at=-1
resume_at=0
conf=0

print_usage() {
    echo "Usage: assembly.sh [flags] file1.fastq fil2.fastq..."
    echo "Flags:"
    echo "  -t NUM_THREADS          Number of threads to use"
    echo "  -s STOP AT              Stop after: 0 flye, 1 medaka, 2 quast, 3 singlem"
    echo "  -r RESUME AT            Resume before:  0 flye, 1 medaka, 2 quast, 3 singlem"
    echo "  -h                      Displays this text"
}

while getopts 't:d:c:s:r:h' flag; do
    case "${flag}" in
        t) num_threads=${OPTARG};;
        d) kraken_db=${OPTARG};;
        c) conf=${OPTARG};;
        s) stop_at=${OPTARG};;
        r) resume_at=${OPTARG};;
        *) print_usage
            exit 1;;
    esac
done

shift $((OPTIND - 1))
fastas=$@

if [ $# -eq 0 ]; then
    echo -e "${red}ERROR:${nocolor} no file provided"
    print_usage
    exit 1
fi

if [ $resume_at -le 0 ]; then 
    echo "Co-assemblying using Flye..."
    mkdir -p assemblies/coassembly/draft/
    flye --nano-hq $fastas --out-dir assemblies/coassembly/draft/ -t $num_threads --meta 
fi

if [ $stop_at -eq 0 ]; then
    exit 0
fi

#if [ $resume_at -le 1 ]; then

    # CHANGE PATHS BELOW TO POINT TO POLISHED FASTA AT polished/consensus.fasta

    # Create a merged fastq
 #   cat $fastqs > .temp_allfastqs.fastq

  #  echo "Polishing co-assembly with Medaka..."
   # mkdir -p assemblies/coassembly/polished/
    #medaka_consensus -i .temp_allfastqs.fastq -b 80 -d assemblies/coassembly/draft/assembly.fasta -o assemblies/coassembly/polished -t 4 -m r1041_e82_400bps_sup_v4.2.0

    #rm .temp_allfastqs.fastq
#fi

if [ $stop_at -eq 1 ]; then
    exit 0
fi

if [ $resume_at -le 2 ]; then
    echo "Evaluating assembly with Metaquast..."
    mkdir -p assemblies/coassembly/quast/
    metaquast -o assemblies/coassembly/quast -t $num_threads assemblies/coassembly/draft/assembly.fasta
fi

if [ $stop_at -eq 2 ]; then
    exit 0;
fi

if [ $resume_at -le 3 ]; then

    # SingleM
    mkdir -p diversity/coassembly/
    singlem pipe --reads assemblies/coassembly/draft/assembly.fasta --threads $num_threads --taxonomic-profile-krona diversity/coassembly/singlem.krona -p diversity/coassembly/singlem-profile.tsv
    ktImportKrona -c -o diversity/coassembly_krona.html diversity/coassembly/singlem.krona
    ktImportKrona -c -o diversity/all_krona.html diversity/*/*.krona diversity/*/*/*.krona
fi
