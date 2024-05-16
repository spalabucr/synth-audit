EXP_NAME=qb_vs_dcr
N_REPS=10000
N_PROCS=32

for EPSILON in 1.0 4.0
do
    for DATA_NAME in adult fire
    do
        for MODEL in DPartPB DSynthPB NIST_MST MST DPWGAN DPWGANCity
        do
            # choose worst possible target for dataset
            if [[ "$DATA_NAME" == "adult" ]]; then
                TARGET_IDX=61
            else
                TARGET_IDX=0
            fi

            # switch between neighbour definition
            if [[ "$MODEL" == "DPartPB" || "$MODEL" == "DSynthPB" ]]; then
                NEIGHBOUR=edit
            else
                NEIGHBOUR=addremove
            fi

            echo "[$(date +%F_%T)] $EPSILON, $DATA_NAME, $MODEL" | tee -a logs/$EXP_NAME.txt
            python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
            --neighbour $NEIGHBOUR --n_procs $N_PROCS --out_dir exp_data/$EXP_NAME/$DATA_NAME --target_idx $TARGET_IDX
        done
    done
done
echo "[$(date +%F_%T)] Done" | tee -a logs/$EXP_NAME.txt
