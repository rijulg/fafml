#!/bin/bash

# Needs imagemagick, libjpeg-progs

for f in **/*.tif **/*.tiff
do  
  basename=${f%.*}
  echo "Converting ${f} TIFF image to JPEG image (compression level 100%)"
  convert "${f}" "${basename}.jpeg"
  djpeg -pnm "${basename}.jpeg" | cjpeg -quality 100 > "${basename}-z80.jpeg"
  chmod 644 "${basename}-z80.jpeg"
done 