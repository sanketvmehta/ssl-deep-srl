#!/bin/bash
#title          : fetch_and_process_OntoNotes.sh
#description    : This script generates *_conll files by populating Co-NLL 2012 *_skel files with data(words) from OntoNotes 5.0. This script also takes care of downloading OntoNotes 5.0, required scripts and data required for generating *_conll files.
#author         : sanketvmehta
#date           : 2018-02-09
#version        : 1.0
#usage          : bash fetch_and_process_OntoNotes.sh
#============================================================================

LDC_FILE=LDC2013T19.tgz
if [ -f $LDC_FILE ]; then
   echo "The file '$LDC_FILE' exists."
else
   echo "Downloading the file '$LDC_FILE'."
   wget {ADD_DATA_URL_HERE}/$LDC_FILE
fi

ONTO_DIR=ontonotes-release-5.0
if [ -d $ONTO_DIR ]; then
   echo "The directory '$ONTO_DIR' exists."
else
   echo "Extracting file '$LDC_FILE' to '$ONTO_DIR' directory."
   tar -xvzf $LDC_FILE
fi

CONLL_FORMAT_ONTO=conll-formatted-ontonotes-5.0
if [ -d $CONLL_FORMAT_ONTO ]; then
   echo "The directory '$CONLL_FORMAT_ONTO' exists."
else
   echo "Downloading the CoNLL formatted ontonotes-5.0 files from the server."
   wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
   tar -xvzf v12.tar.gz
   mv $CONLL_FORMAT_ONTO-12/$CONLL_FORMAT_ONTO $CONLL_FORMAT_ONTO
   rm -r $CONLL_FORMAT_ONTO-12
fi

CONLL_FORMAT_ONTO_SCRIPTS=conll-formatted-ontonotes-5.0/scripts
CONLL_FORMAT_ONTO_SCRIPTS_TAR=conll-formatted-ontonotes-5.0-scripts.tar.gz
if [ -d $CONLL_FORMAT_ONTO_SCRIPTS ]; then
   echo "The directory '$CONLL_FORMAT_ONTO_SCRIPTS' exists."
else
   echo "Downloading and running scripts to convert *_skel files to *_conll files."
   wget http://ontonotes.cemantix.org/download/$CONLL_FORMAT_ONTO_SCRIPTS_TAR
   tar -xvzf $CONLL_FORMAT_ONTO_SCRIPTS_TAR

   CURR_DIR=$PWD
   echo $CURR_DIR
   cd $CONLL_FORMAT_ONTO_SCRIPTS
   sed -i 's/python/python2.7/g' skeleton2conll.sh
   bash skeleton2conll.sh -D $CURR_DIR/ontonotes-release-5.0/data/files/data $CURR_DIR/$CONLL_FORMAT_ONTO
fi
