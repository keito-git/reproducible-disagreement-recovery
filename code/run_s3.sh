export PIP_CONFIG_FILE=/dev/null
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; R=../results/s3
[ -f $SP/splits.json ] || HF_DATASETS_OFFLINE=1 $PY s3_judge.py split --votes $VT --out $SP
for D in brighter_eng goemotions; do
  for M in microsoft/deberta-v3-large roberta-large; do for LR in 1e-5 2e-5; do
    O=$R/$D/sel/$(basename $M)_lr$LR
    [ -f $O/info.json ] || $PY s3_judge.py train --votes $VT --split $SP --data $D --model $M --lr $LR --seed 0 --out $O
  done; done
  BEST=$($PY -c "import json,glob; r=[(json.load(open(f)),f.split('/')[-2]) for f in glob.glob('$R/$D/sel/*/info.json')]; b=min(r,key=lambda x:x[0]['dev_jsd']); print(b[0]['model'],b[0]['lr'],b[1])")
  echo "$D best $BEST"; set -- $BEST
  mkdir -p $R/$D/main; [ -d $R/$D/main/seed0 ] || cp -r $R/$D/sel/$3 $R/$D/main/seed0
  for S in 1 2; do [ -f $R/$D/main/seed$S/info.json ] || $PY s3_judge.py train --votes $VT --split $SP --data $D --model $1 --lr $2 --seed $S --out $R/$D/main/seed$S; done
  if [ $D = brighter_eng ]; then for F in 0.25 0.5; do for S in 0 1 2; do
    O=$R/$D/frac$F/seed$S; [ -f $O/info.json ] || $PY s3_judge.py train --votes $VT --split $SP --data $D --model $1 --lr $2 --seed $S --frac $F --out $O
  done; done; fi
done
echo S3_DONE
