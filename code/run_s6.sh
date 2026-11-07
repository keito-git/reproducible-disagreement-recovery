export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_CONFIG_FILE=/dev/null
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; RAW=../data/raw; R=../results/s6
J="$PY s4_judge.py train --votes $VT --raw $RAW --split $SP --data brighter_eng --model roberta-large --lr 2e-5 --ntrain 1868"
for S in 0 1 2; do
  O=$R/brighter_eng/n1868_thin7/seed$S; [ -f $O/info.json ] || $J --seed $S --thin 7 --out $O
  O=$R/brighter_eng/n1868_full/seed$S; [ -f $O/info.json ] || $J --seed $S --out $O
done
echo S6_DONE
