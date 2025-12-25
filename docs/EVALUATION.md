# Benchmark

## CASP15 Benchmark

1. Transfer benchmark data
  ```bash
  cp /mnt/parallel_storage/wykim_lab/icl_shwan/data/benchmark.tar.zst ./
  tar -xvf benchmark.tar.zst -C ./benchmark/
  ```

2. Run inference (sampling)
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/inference_multigpu.py \
    --config {CONFIG_PATH} \
    --checkpoint {CKPT_PATH} \
    -i ./benchmark/casp15/inputs/queries/ \
    -o ./benchmark/casp15/predictions/kfold/
  ```

3. Evaluation
  ```bash
  cd ./benchmark/

  # Run evaluation using openstructure.
  module load containers/apptainer/1.3.4
  bash eval_ost.sh

  # Aggregate results and plot the results.
  python ./scripts/aggr_results.py
  ```

