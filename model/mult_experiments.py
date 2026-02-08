import csv
import os

from tsl import logger
from tsl.experiment import Experiment
from run_exp import run_imputation

if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    print(exp)
    res = exp.run()
    logger.info(res)
    
    shift = res.pop('spatial')
    eval_setting = res.pop('eval_setting')
    node_features = res.pop('node_f')
    base_dir = os.path.join("res")

    if shift:
        filename = f"{res['model']}-{node_features}.csv"
    else:
        filename = f"{res['model']}-RND.csv"

    csv_path = os.path.join(base_dir, filename)
    file_exists = os.path.exists(csv_path)

    os.makedirs(base_dir, exist_ok=True)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(res)