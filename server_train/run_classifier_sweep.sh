#!/usr/bin/env bash

# Run the complete 72-combination classifier sweep sequentially.
# Launch this script with nohup after activating the snn environment.

set -u

PROJECT_ROOT="${PROJECT_ROOT:-/Data0/kevinswk/sw_train}"
DATASET_PATH="${DATASET_PATH:-/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/trained_models/classifier_sweep}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

TEMPERATURES=(0.1 0.2 0.5 1.0)
TIME_STEPS=(15 20 30)
OSCILLATORS=(90 50 30)
EMBEDDING_DIMS=(16 20)

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}" || exit 1

total=72
experiment_index=0

for temperature in "${TEMPERATURES[@]}"; do
    temperature_tag="${temperature/./p}"
    for time_steps in "${TIME_STEPS[@]}"; do
        for num_oscillators in "${OSCILLATORS[@]}"; do
            for embedding_dim in "${EMBEDDING_DIMS[@]}"; do
                experiment_index=$((experiment_index + 1))
                run_name="temp${temperature_tag}_T${time_steps}_N${num_oscillators}_D${embedding_dim}"
                run_dir="${OUTPUT_ROOT}/${run_name}"
                visualization_dir="${run_dir}/visualizations"

                mkdir -p "${run_dir}"
                {
                    echo "classifier_temperature=${temperature}"
                    echo "num_feature_maps=${time_steps}"
                    echo "num_regions=${num_oscillators}"
                    echo "classifier_embedding_dim=${embedding_dim}"
                } > "${run_dir}/experiment_config.txt"

                echo "[$(date '+%F %T')] [${experiment_index}/${total}] ${run_name}"

                if [[ -f "${run_dir}/COMPLETED" ]]; then
                    echo "Already completed; skipping ${run_name}"
                    continue
                fi

                if [[ ! -f "${run_dir}/TRAINING_COMPLETED" ]]; then
                    if python -u -m snn_kuramoto_bidirectional.training.train_s2net_autoencoder \
                        --dataset-path "${DATASET_PATH}" \
                        --hdf5-key image \
                        --output-dir "${run_dir}" \
                        --num-images 1000 \
                        --image-size 128 \
                        --validation-fraction 0.1 \
                        --num-feature-maps "${time_steps}" \
                        --num-objects 11 \
                        --num-oscillators "${num_oscillators}" \
                        --classifier-temperature "${temperature}" \
                        --classifier-embedding-dim "${embedding_dim}" \
                        --kernel-size 3 \
                        --epochs 200 \
                        --batch-size 8 \
                        --learning-rate 1e-3 \
                        --weight-decay 0 \
                        --gradient-clip 1.0 \
                        --sc-momentum 0.99 \
                        --num-workers 0 \
                        --checkpoint-every 200 \
                        --preview-every 100000 \
                        --preview-images 3 \
                        --device cuda \
                        > "${run_dir}/training.log" 2>&1
                    then
                        touch "${run_dir}/TRAINING_COMPLETED"
                    else
                        echo "[$(date '+%F %T')] TRAINING FAILED: ${run_name}" \
                            | tee -a "${OUTPUT_ROOT}/failed_experiments.log"
                        continue
                    fi
                fi

                if python -u -m snn_kuramoto_bidirectional.visualization \
                    --checkpoint-path "${run_dir}/best.pt" \
                    --dataset-path "${DATASET_PATH}" \
                    --hdf5-key image \
                    --output-dir "${visualization_dir}" \
                    --start-index 0 \
                    --num-images 3 \
                    --device cuda \
                    > "${run_dir}/visualization.log" 2>&1
                then
                    touch "${run_dir}/COMPLETED"
                    echo "[$(date '+%F %T')] COMPLETED: ${run_name}"
                else
                    echo "[$(date '+%F %T')] VISUALIZATION FAILED: ${run_name}" \
                        | tee -a "${OUTPUT_ROOT}/failed_experiments.log"
                fi
            done
        done
    done
done

echo "[$(date '+%F %T')] Sweep finished."
