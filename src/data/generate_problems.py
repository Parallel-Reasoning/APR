"""
Script for generating core countdown problems (target/nums/solutions).
"""
import json
import argparse
import random
import os
import multiprocessing as mp
from functools import partial
import tqdm

from src.data.countdown import CountDown


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--data_dir", type=str, default=None, help="Directory to store data")
parser.add_argument("--start_range", type=int, default=3, help="Max Range of starting numbers [M, N]")
parser.add_argument("--min_range", type=int, default=3, help="Min Range of starting numbers [M, N]")
parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of data samples to generate")
parser.add_argument("--grow", action="store_true", help="grow mode on or off, only a new train set is created")
parser.add_argument("--train_only", action="store_true", help="only generate train set")
parser.add_argument("--offset", type=int, default=1, help="offset for random seed")
parser.add_argument("--num_workers", type=int, default=int(mp.cpu_count()/4), help="Number of worker processes")
parser.add_argument("--existing_problems", type=str, default=None, help="Path to existing problems")


def generate_problem(args, target_nums, split=None, train_nums_targets=None, _=None):
    start_size = random.randint(args.min_range, args.start_range)
    cd = CountDown(args.max_target, start_size)
    
    target = random.choice(target_nums)
    nums, solution = cd.generate(target)
    no_backtrack_trace = cd.convert_to_path(target, nums, solution)
    
    if train_nums_targets:
        # Create a string hash of sorted nums and target
        nums_target_hash = f"{sorted(nums)}_{target}"
        while nums_target_hash in train_nums_targets:
            target = random.choice(target_nums)
            nums, solution = cd.generate(target)
            no_backtrack_trace = cd.convert_to_path(target, nums, solution)
            nums_target_hash = f"{sorted(nums)}_{target}"
            
    return {
        "nums": nums,
        "target": target,
        "solution": solution,
        "optimal_path": no_backtrack_trace,
        "start_size": start_size
    }


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Using {args.num_workers} CPU workers")
    random.seed(args.seed)
    
    target_nums = [i for i in range(10, args.max_target+1)]
    train_nums = target_nums

    if args.grow:
        splits = ["grow"]
        target_list = [train_nums]
        random.seed(args.seed + args.offset)
    elif args.train_only:
        splits = ["train"]
        target_list = [train_nums]
    else:
        splits = ["train", "val"]
        target_list = [train_nums, train_nums]

    data_samples = {}
    if args.existing_problems:
        print(f"Loading existing problems from {args.existing_problems}")
        with open(args.existing_problems, "r") as f:
            existing_problems = json.load(f)
        print(f"Loaded {len(existing_problems)} existing problems")
    
    for split, target_nums in zip(splits, target_list):
        print(f"Generating {split} set")
        data_samples[split] = []
        num_samples = args.num_samples if split in ["train", "grow"] else 1000

        train_nums_targets = None
        if split == "val":
            # Create a set of string hashes for fast lookup
            train_nums_targets = set(f"{sorted(s['nums'])}_{s['target']}" for s in data_samples["train"])
        elif args.existing_problems:
            train_nums_targets = set(f"{sorted(s['nums'])}_{s['target']}" for s in existing_problems)
        elif split in ["train", "grow"]:
            # Initialize empty set to track duplicates within the training set itself
            train_nums_targets = set()

        # Generate problems
        pool = mp.Pool(args.num_workers)
        gen_problem = partial(generate_problem, args, target_nums, split, train_nums_targets)
        samples = list(tqdm.tqdm(pool.imap(gen_problem, range(num_samples)), 
                               total=num_samples, 
                               desc=f"Generating {split}"))
        
        pool.close()
        pool.join()

        data_samples[split].extend(samples)

        os.makedirs(args.data_dir, exist_ok=True)
        output_path = f"{args.data_dir}/{split}{args.offset}_b{args.start_range}_t{args.max_target}_n{args.num_samples}_problems.json"
        with open(output_path, "w") as f:
            json.dump(data_samples[split], f, indent=4) 