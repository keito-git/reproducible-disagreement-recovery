export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_CONFIG_FILE=/dev/null
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; RAW=../data/raw; R=../results/s4
J="$PY s4_judge.py train --votes $VT --raw $RAW --split $SP"
[ -f $SP/splits_e2.json ] || $PY s4_judge.py split --votes $VT --raw $RAW --out $SP
# ChaosNLI: selection on dev, then 3 seeds
for M in roberta-large FacebookAI/roberta-large-mnli microsoft/deberta-v3-large; do for LR in 1e-5 2e-5; do
  O=$R/chaosnli/sel/$(basename $M)_lr$LR
  [ -f $O/info.json ] || $J --data chaosnli --model $M --lr $LR --seed 0 --out $O
done; done
BEST=$($PY -c "import json,glob; r=[(json.load(open(f)),f.split('/')[-2]) for f in glob.glob('$R/chaosnli/sel/*/info.json')]; b=min(r,key=lambda x:x[0]['dev_jsd']); print(b[0]['model'],b[0]['lr'],b[1])")
echo "chaosnli best $BEST"; set -- $BEST
mkdir -p $R/chaosnli/main; [ -d $R/chaosnli/main/seed0 ] || cp -r $R/chaosnli/sel/$3 $R/chaosnli/main/seed0
for S in 1 2; do O=$R/chaosnli/main/seed$S; [ -f $O/info.json ] || $J --data chaosnli --model $1 --lr $2 --seed $S --out $O; done
# DICES-990: 5-fold cross-fitting, configuration fixed (roberta-large, 2e-5)
for F in 0 1 2 3 4; do for S in 0 1 2; do
  O=$R/dices_990/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_990 --model roberta-large --lr 2e-5 --seed $S --fold $F --out $O
done; done
# BRIGHTER English vote thinning (configuration selected in run_s3.sh)
for K in 1 3; do for S in 0 1 2; do
  O=$R/brighter_eng/thin$K/seed$S; [ -f $O/info.json ] || $J --data brighter_eng --model roberta-large --lr 2e-5 --seed $S --thin $K --out $O
done; done
echo S4_DONE
