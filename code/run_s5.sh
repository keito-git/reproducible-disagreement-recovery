export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_CONFIG_FILE=/dev/null
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; RAW=../data/raw; R=../results/s5
J="$PY s4_judge.py train --votes $VT --raw $RAW --split $SP"
MN=FacebookAI/roberta-large-mnli
for S in 0 1 2; do
  for K in 7 3; do O=$R/chaosnli/mnli_thin$K/seed$S; [ -f $O/info.json ] || $J --data chaosnli --model $MN --lr 2e-5 --seed $S --thin $K --out $O; done
  O=$R/chaosnli/rl_thin7/seed$S; [ -f $O/info.json ] || $J --data chaosnli --model roberta-large --lr 2e-5 --seed $S --thin 7 --out $O
  O=$R/chaosnli/rl_full/seed$S
  if [ $S = 0 ]; then mkdir -p $R/chaosnli/rl_full; [ -d $O ] || cp -r ../results/s4/chaosnli/sel/roberta-large_lr2e-5 $O
  else [ -f $O/info.json ] || $J --data chaosnli --model roberta-large --lr 2e-5 --seed $S --out $O; fi
done
for K in 7 3; do for F in 0 1 2 3 4; do for S in 0 1 2; do
  O=$R/dices_990/thin$K/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_990 --model roberta-large --lr 2e-5 --seed $S --fold $F --thin $K --out $O
done; done; done
echo S5_DONE
