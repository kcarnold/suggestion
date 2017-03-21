#!/bin/sh
NAME=models/$1
KENLM=../kenlm/build/bin
$KENLM/lmplz -o 5 --prune 2  --verbose_header < $NAME.txt > $NAME.arpa
$KENLM/build_binary $NAME.arpa $NAME.kenlm
