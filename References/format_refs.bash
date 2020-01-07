#!/bin/bash

PANDOC="/home/ctroupin/bin/pandoc"
PANDOCRUN="${PANDOC} -t markdown_strict --filter=pandoc-citeproc --standalone --csl ocean-science.csl"

for reffiles in $(ls ./*ref.md); do
		outputfile=${reffiles%'_ref.md'}".md"
		echo "Working on ${reffiles} --> ${outputfile}"
		${PANDOCRUN} ${reffiles} -o ${outputfile}
done

echo "Done"
