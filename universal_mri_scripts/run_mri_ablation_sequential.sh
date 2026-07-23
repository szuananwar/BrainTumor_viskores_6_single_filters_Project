#!/bin/bash

DATA_ROOT=$1
OUT_ROOT=$2
CLASSES_STRING=$3

if [ -z "$DATA_ROOT" ] || [ -z "$OUT_ROOT" ] || [ -z "$CLASSES_STRING" ]; then
    echo "Usage:"
    echo "bash universal_mri_scripts/run_mri_ablation_sequential.sh DATA_ROOT OUT_ROOT \"Class1 Class2 Class3\""
    exit 1
fi

cd /lustre/orion/med141/scratch/sn1/brain_tumor_viskores/project

mkdir -p logs
mkdir -p "$OUT_ROOT"

COMBOS=(
single_median
single_sobel
single_laplacian
single_texture
single_otsu
single_morphology
pair_median_sobel
pair_sobel_laplacian
pair_texture_otsu
pair_otsu_morphology
triple_median_sobel_laplacian
triple_texture_otsu_morphology
triple_sobel_texture_morphology
all_features
)

for COMBO in "${COMBOS[@]}"
do
    echo "Submitting $COMBO"

    SLURM_FILE="logs/train_${COMBO}.slurm"

    cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH -A med141
#SBATCH -J ${COMBO}
#SBATCH -o logs/${COMBO}_%j.out
#SBATCH -e logs/${COMBO}_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

cd /lustre/orion/med141/scratch/sn1/brain_tumor_viskores/project

source /lustre/orion/med141/scratch/sn1/brain_tumor_viskores/envs/torch_rocm/bin/activate
export TORCH_HOME=/lustre/orion/med141/scratch/sn1/brain_tumor_viskores/torch_cache
export MIOPEN_DISABLE_CACHE=1

python universal_mri_scripts/train_mri_vit.py \
--data ${DATA_ROOT}/${COMBO} \
--output ${OUT_ROOT}/${COMBO} \
--classes ${CLASSES_STRING} \
--epochs 20 \
--batch_size 16
EOF

    JOBID=$(sbatch "$SLURM_FILE" | awk '{print $4}')
    echo "Submitted job $JOBID for $COMBO"

    while squeue -j "$JOBID" | grep -q "$JOBID"; do
        sleep 60
    done

    echo "$COMBO finished."
    COUNT=$(find "$OUT_ROOT" -name "metrics.json" | wc -l)
    echo "Completed metrics files so far: $COUNT"
done

echo "All ablation jobs finished."
