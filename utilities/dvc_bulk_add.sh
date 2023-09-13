#!/bin/bash
#usage: bash dvc_bulk_add.sh

#specify path to dvc and git add all files
path='/Users/william.nadolski/Desktop/Git/acs-ampac/data/output'
echo Performing DVC add for all files within specified directory: 
echo $path

#loop through files and dvc/git add each
#skip over subfolder and existing .dvc files
cd $path 
search_dir=$path
for entry in "$search_dir"/*
do
  if [ -f "$entry" ] && [ "${entry##*.}" != "dvc" ];then
  echo DVC Add: "$entry"
  dvc add "$entry"
  echo Git Add: "$entry".dvc
  git add "$entry".dvc
  fi
done

