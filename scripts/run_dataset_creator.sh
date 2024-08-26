#!/bin/bash

for i in {0..3}
do
    sbatch /home/iscb/wolfson/omriyakir/cpp_deep/scripts/submit_job_dataset_creator.sh $i
done
