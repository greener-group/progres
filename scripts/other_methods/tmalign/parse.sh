rm run_unsorted.out run.out
for file in out/*.out; do
  echo $file
  grep -a -n "^Name of Chain_1:\|^Name of Chain_2:\|TM-score=" $file | sed 's|=|:|g' | awk -F ": " '{print $2}' | awk '{print $1}' | sed 's|../pdbstyle-2.08/||g' | xargs -n 4 | awk '{print $1, $2, $3, $4}' >> run_unsorted.out
done
sort -k1b,1 -nrk3,3 run_unsorted.out > run.out
