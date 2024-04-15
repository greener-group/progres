mkdir out
while read query; do
  echo $query
  for target in ../pdbstyle-2.08/*; do
    TMalign ../pdbstyle-2.08/$query.ent $target -fast >> out/$query.out
  done
done < domids.txt
