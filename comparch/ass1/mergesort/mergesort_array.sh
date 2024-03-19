#!/bin/sh
SORT="mergesort"
DIR="./${SORT}_diffarray"
if [ -d "$DIR" ]; then
   echo "Directory '$DIR' found"
else
   mkdir "$DIR"
   echo "Directory '$DIR' created"
fi

g++ "${SORT}_250.c" -O2 -g -o "$DIR/${SORT}_250.out"
valgrind --tool=cachegrind --cachegrind-out-file="$DIR/${SORT}_250.out.cachegrind" "$DIR/${SORT}_250.out" 2> "$DIR/${SORT}_250_output.txt"
cg_annotate "$DIR/${SORT}_250.out.cachegrind" > "$DIR/${SORT}_250_annotate_output.txt"
echo "\n${SORT} - Array of length 250000 sorted"
echo "Output stored in $DIR/${SORT}_250_output.txt and $DIR/${SORT}_250_annotate_output.txt"

g++ "${SORT}_500.c" -O2 -g -o "$DIR/${SORT}_500.out"
valgrind --tool=cachegrind --cachegrind-out-file="$DIR/${SORT}_500.out.cachegrind" "$DIR/${SORT}_500.out" 2> "$DIR/${SORT}_500_output.txt"
cg_annotate "$DIR/${SORT}_500.out.cachegrind" > "$DIR/${SORT}_500_annotate_output.txt"
echo "\n${SORT} - Array of length 500000 sorted"
echo "Output stored in $DIR/${SORT}_500_output.txt and $DIR/${SORT}_500_annotate_output.txt"


g++ "${SORT}_750.c" -O2 -g -o "$DIR/${SORT}_750.out"
valgrind --tool=cachegrind --cachegrind-out-file="$DIR/${SORT}_750.out.cachegrind" "$DIR/${SORT}_750.out" 2> "$DIR/${SORT}_750_output.txt"
cg_annotate "$DIR/${SORT}_750.out.cachegrind" > "$DIR/${SORT}_750_annotate_output.txt"
echo "\n${SORT} - Array of length 750000 sorted"
echo "Output stored in $DIR/${SORT}_750_output.txt and $DIR/${SORT}_750_annotate_output.txt"


g++ "${SORT}_1000.c" -O2 -g -o "$DIR/${SORT}_1000.out"
valgrind --tool=cachegrind --cachegrind-out-file="$DIR/${SORT}_1000.out.cachegrind" "$DIR/${SORT}_1000.out" 2> "$DIR/${SORT}_1000_output.txt"
cg_annotate "$DIR/${SORT}_1000.out.cachegrind" > "$DIR/${SORT}_1000_annotate_output.txt"
echo "\n${SORT} - Array of length 1000000 sorted"
echo "Output stored in $DIR/${SORT}_1000_output.txt and $DIR/${SORT}_1000_annotate_output.txt"
echo "\n\n"

touch "comparison_diff_arraylength.txt"
echo "\n" > "comparison_diff_arraylength.txt"
sed -n '2,4p' "$DIR/${SORT}_1000_annotate_output.txt" > "comparison_diff_arraylength.txt"

echo "\n\nArray Length: 250000" >> "comparison_diff_arraylength.txt"
sed -n '14,18p' "$DIR/${SORT}_250_annotate_output.txt" >> "comparison_diff_arraylength.txt"

echo "\n\nArray Length: 500000" >> "comparison_diff_arraylength.txt"
sed -n '14,18p' "$DIR/${SORT}_500_annotate_output.txt" >> "comparison_diff_arraylength.txt"

echo "\n\nArray Length: 750000" >> "comparison_diff_arraylength.txt"
sed -n '14,18p' "$DIR/${SORT}_750_annotate_output.txt" >> "comparison_diff_arraylength.txt"

echo "\n\nArray Length: 1000000" >> "comparison_diff_arraylength.txt"
sed -n '14,18p' "$DIR/${SORT}_1000_annotate_output.txt" >> "comparison_diff_arraylength.txt"

cat "comparison_diff_arraylength.txt"
