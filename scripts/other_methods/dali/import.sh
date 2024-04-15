mkdir imported
while read domid_pdbid; do
  domid=$(echo $domid_pdbid | cut -f1 -d " ")
  pdbid=$(echo $domid_pdbid | cut -f2 -d " ")
  echo $domid
  perl ~/soft/DaliLite.v5/bin/import.pl --pdbfile ../pdbstyle-2.08/$domid.ent --pdbid mol1 --dat ./ --clean
  mv mol1?.dat imported/${pdbid}A.dat
done < domids_pdbids.txt
