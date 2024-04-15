mkdir run
while read domid; do
  echo $domid
  mkdir $domid
  cd $domid
  perl ~/soft/DaliLite.v5/bin/dali.pl --pdbfile1 ../../pdbstyle-2.08/$domid.ent --db ../pdbids_imported.txt --dat1 ./ --dat2 ../imported --clean
  mv mol1?.txt ../run/$domid.txt
  cd ..
  rm -r $domid
done < domids_imported.txt
