#!/bin/bash
#title          : eval.sh
#description    : This script evaluates the predictions using srl-eval.pl program which is the official script for evaluation provided in CoNLL-2005 shared task.
#author         : sanketvmehta
#date           : 2018-02-13
#version        : 1.0
#usage          : bash eval.sh <path-to-gold-labels-file> <path-to-predicted-labels-file>
#============================================================================

CURR_DIR=$PWD
echo $CURR_DIR
MAIN_DIR="$(dirname "$CURR_DIR")"

SRL_CONLL_PATH=$MAIN_DIR/data/srlconll-1.1

if [ -d $SRL_CONLL_PATH ]; then
    echo "The directory '$SRL_CONLL_PATH' exists."
else
    echo "Downloading and extracting file from the server!"
    wget -P $MAIN_DIR/data/ http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
    tar -xvzf $MAIN_DIR/data/srlconll-1.1.tgz -C $MAIN_DIR/data/
fi

echo "Using the official script (CoNLL-2005) for evaluation!"

export PERL5LIB="$SRL_CONLL_PATH/lib:$PERL5LIB"
export PATH="$SRL_CONLL_PATH/bin:$PATH"

GOLD_FILE_PATH=$1
PREDICTED_FILE_PATH=$2

perl $SRL_CONLL_PATH/bin/srl-eval.pl $GOLD_FILE_PATH $PREDICTED_FILE_PATH