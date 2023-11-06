#!/bin/bash


python3 ./train.py \
--transcription_factor MAX \
--train A549 \
--valid GM12878 \
--gpu 3 \
--name leopard > leopard.log