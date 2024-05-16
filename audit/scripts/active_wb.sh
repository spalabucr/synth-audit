EXP_NAME=active_wb
N_REPS=10000
N_PROCS=32
DATA_NAME=adult

MODEL=DPWGAN
if [[ "$MODEL" == "DPartPB" || "$MODEL" == "DSynthPB" ]]; then
    NEIGHBOUR=edit
else
    NEIGHBOUR=addremove
fi

for EPSILON in 1.0 2.0 4.0 10.0
do
    echo "[$(date +%F_%T)] $MODEL, $EPSILON" | tee -a logs/$EXP_NAME.txt
    python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
    --neighbour $NEIGHBOUR --worstcase --repeat --active_wb --n_procs $N_PROCS \
    --out_dir exp_data/$EXP_NAME/$DATA_NAME --save_model_only

    echo "[$(date +%F_%T)] $MODEL, $EPSILON (LOGAN)" | tee -a logs/$EXP_NAME.txt
    python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
    --neighbour $NEIGHBOUR --worstcase --repeat --n_procs $N_PROCS \
    --out_dir exp_data/$EXP_NAME/$DATA_NAME
done
echo "[$(date +%F_%T)] Done" | tee -a logs/$EXP_NAME.txt
