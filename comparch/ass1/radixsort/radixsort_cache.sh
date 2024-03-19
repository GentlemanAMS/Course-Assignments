#!/bin/sh

SORT="radixsort"
DIR="${SORT}_diffconfig"
if [ -d "$DIR" ]; then
   echo "\nDirectory '$DIR' found"
else
   mkdir "$DIR"
   echo "\nDirectory '$DIR' created"
fi

echo "\nArray Size: 500000"
g++ "${SORT}.c" -O2 -g -o "$DIR/${SORT}.out"
valgrind --tool=cachegrind --cachegrind-out-file="$DIR/${SORT}_default.out.cachegrind" "$DIR/${SORT}.out" 2> "$DIR/${SORT}_default_output.txt"
cg_annotate "$DIR/${SORT}_default.out.cachegrind" > "$DIR/${SORT}_default_annotate_output.txt" 
echo "\nDefault Cache Configuration: Sorted"
echo "Output stored in $DIR/${SORT}_default_annotate_output.txt and $DIR/${SORT}_default_output.txt"

valgrind --tool=cachegrind --D1=49152,24,64 --cachegrind-out-file="$DIR/${SORT}_config1.out.cachegrind" "$DIR/${SORT}.out" 2> "$DIR/${SORT}_config1_output.txt"
cg_annotate "$DIR/${SORT}_config1.out.cachegrind" > "$DIR/${SORT}_config1_annotate_output.txt"
echo "\nCache Configuration 1 : Sorted"
echo "Output stored in $DIR/${SORT}_config1_annotate_output.txt and $DIR/${SORT}_config1_output.txt"

valgrind --tool=cachegrind --D1=49152,6,64 --cachegrind-out-file="$DIR/${SORT}_config2.out.cachegrind" "$DIR/${SORT}.out" 2> "$DIR/${SORT}_config2_output.txt"
cg_annotate "$DIR/${SORT}_config2.out.cachegrind" > "$DIR/${SORT}_config2_annotate_output.txt"
echo "\nCache Configuration 2 : Sorted"
echo "Output stored in $DIR/${SORT}_config2_annotate_output.txt and $DIR/${SORT}_config2_output.txt"

valgrind --tool=cachegrind --LL=27262976,26,64 --cachegrind-out-file="$DIR/${SORT}_config3.out.cachegrind" "$DIR/${SORT}.out" 2> "$DIR/${SORT}_config3_output.txt"
cg_annotate "$DIR/${SORT}_config3.out.cachegrind" > "$DIR/${SORT}_config3_annotate_output.txt"
echo "\nCache Configuration 3 : Sorted"
echo "Output stored in $DIR/${SORT}_config3_annotate_output.txt and $DIR/${SORT}_config3_output.txt"

valgrind --tool=cachegrind --LL=27262976,52,64 --cachegrind-out-file="$DIR/${SORT}_config4.out.cachegrind" "$DIR/${SORT}.out" 2> "$DIR/${SORT}_config4_output.txt"
cg_annotate "$DIR/${SORT}_config4.out.cachegrind" > "$DIR/${SORT}_config4_annotate_output.txt"
echo "\nCache Configuration 4 : Sorted"
echo "Output stored in $DIR/${SORT}_config4_annotate_output.txt and $DIR/${SORT}_config4_output.txt"
echo "\n\n"

touch "comparison_diff_config.txt"
echo "\n" > "comparison_diff_config.txt"

echo "\n\nDefault Configuration" >> "comparison_diff_config.txt"
sed -n '2,4p' "$DIR/${SORT}_default_annotate_output.txt" >> "comparison_diff_config.txt"
#echo "\n" >> "comparison_diff_config.txt"
sed -n '14,18p' "$DIR/${SORT}_default_annotate_output.txt" >> "comparison_diff_config.txt"

echo "\n\nConfiguration 1" >> "comparison_diff_config.txt"
sed -n '2,4p' "$DIR/${SORT}_config1_annotate_output.txt" >> "comparison_diff_config.txt"
#echo "\n" >> "comparison_diff_config.txt"
sed -n '14,18p' "$DIR/${SORT}_config1_annotate_output.txt" >> "comparison_diff_config.txt"

echo "\n\nConfiguration 2" >> "comparison_diff_config.txt"
sed -n '2,4p' "$DIR/${SORT}_config2_annotate_output.txt" >> "comparison_diff_config.txt"
#echo "\n" >> "comparison_diff_config.txt"
sed -n '14,18p' "$DIR/${SORT}_config2_annotate_output.txt" >> "comparison_diff_config.txt"

echo "\n\nConfiguration 3" >> "comparison_diff_config.txt"
sed -n '2,4p' "$DIR/${SORT}_config3_annotate_output.txt" >> "comparison_diff_config.txt"
#echo "\n" >> "comparison_diff_config.txt"
sed -n '14,18p' "$DIR/${SORT}_config3_annotate_output.txt" >> "comparison_diff_config.txt"

echo "\n\nConfiguration 4" >> "comparison_diff_config.txt"
sed -n '2,4p' "$DIR/${SORT}_config4_annotate_output.txt" >> "comparison_diff_config.txt"
#echo "\n" >> "comparison_diff_config.txt"
sed -n '14,18p' "$DIR/${SORT}_config4_annotate_output.txt" >> "comparison_diff_config.txt"

cat "comparison_diff_config.txt"
