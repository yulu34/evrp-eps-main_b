import os
import subprocess
import argparse
from utils.util import set_device
from eval import eval

def write_metrics_to_csv(metrics, output_dir, filename="metrics.csv"):
    """
    Write metrics to a CSV file.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics
    output_dir : str
        Output directory path
    filename : str, optional
        Name of the CSV file, by default "metrics.csv"
    """
    import os
    import csv
    import datetime
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the full path
    csv_path = os.path.join(output_dir, filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)
    
    # Add timestamp to metrics
    metrics_copy = metrics.copy()  # 创建副本以避免修改原始字典
    metrics_copy['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Define the field names (headers) aligned with eval.py's summary dictionary keys
    fieldnames = [
        'timestamp',
        # Basic metrics
        'avg_obj',
        'std_obj',
        'avg_tour_length',
        'std_tour_length',
        'avg_conflict_cost',
        'std_conflict_cost',
        'avg_actual_tour_length',
        'std_actual_tour_length',
        'total_calc_time',
        'avg_calc_time',
        'std_calc_time',
        # Service metrics
        'percent_served',
        'avg_first_response',
        # Utilization metrics
        'vehicle_utilization',
        'avg_travel_time',
        'avg_wait_time',
        'avg_station_utilization',
        # Queue metrics
        'avg_queue_length',
        'max_queue_length',
        'avg_queue_time',
        'max_queue_time',
        # Charging event metrics
        'avg_charge_events',
        'avg_charge_per_event',
        # Energy metrics
        'total_travel_energy',
        'total_charge_energy',
        #'total_supply_energy'
    ]
    
    # Add any other fields that might be in the metrics dict but not predefined
    existing_keys = list(fieldnames)  # Start with aligned keys
    for key in metrics_copy.keys():
        if key not in existing_keys:
            fieldnames.append(key)  # Append any extra keys found
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header if file didn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics, using 'N/A' for missing keys
        row_data = {k: metrics_copy.get(k, 'N/A') for k in fieldnames}
        writer.writerow(row_data)
    
    print(f"Metrics saved to {csv_path}")

def valid(args: argparse.Namespace) -> None:
    # compare each epoch on validation datasets
    best_epoch = 0
    min_cost   = 1e+9 # a large value

    for epoch in range(args.max_epoch+1): 
        print(f"Evaluating the model at epoch{epoch} (currently best epoch is {best_epoch}: cost={min_cost})", flush=True)
        # load a trained model
        model_path = f"{args.model_dir}/model_epoch{epoch}.pth"
        
        try:
            res = eval(dataset_path=args.dataset_path,
                      eval_batch_size=args.eval_batch_size,
                      model_type="rl",
                      model_path=model_path,
                      decode_type="greedy",
                      conflict_coef=args.conflict_coef,
                      vehicle_speed=args.vehicle_speed,
                      wait_time=args.wait_time,
                      time_horizon=args.time_horizon,
                      random_seed=1234,
                      gpu=args.gpu,
                      num_workers=args.num_workers)
            cost = res["avg_obj"]
            write_metrics_to_csv(res, args.output_dir)
        except AssertionError as e:
            if "there is no node that the vehicle can visit" in str(e):
                print(f"Skipping epoch {epoch} due to no valid nodes error")
                cost = float('inf')  # Set cost to infinity to avoid selecting this model
            else:
                raise e
            
        # if the current epoch is better than previous epochs
        if min_cost > cost:
            best_epoch = epoch
            min_cost   = cost

    # save the best epoch
    model_path = f"{args.model_dir}/model_epoch{best_epoch}.pth"
    save_path  = f"{args.model_dir}/model_bestepoch.pth"
    subprocess.run(f"cp {model_path} {save_path}", shell=True)


if __name__ == "__main__":
    import datetime
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--gpu",              type=int, default=-1)
    parser.add_argument("--num_workers",      type=int, default=8) # 4-》8
    parser.add_argument("--output_dir",       type=str, default=f"results/results_{now.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--log_fname",        type=str, default=None)

    # dataset settings
    parser.add_argument("--dataset_path",    type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256) 

    # model settings
    parser.add_argument("--model_dir",    type=str,   required=True)
    #parser.add_argument("--penalty_coef", type=float, default=100)
    parser.add_argument("--conflict_coef", type=float, default=10.0, help="充电站冲突成本系数")
    
    parser.add_argument("--max_epoch",    type=int,   default=0) # 我是训练了两轮 readme里做的时候这里需要1-1 =0 才能运行

    # other parameters
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_fname is not None:
        os.makedirs(os.path.dirname(args.log_fname), exist_ok=True)
    valid(args)