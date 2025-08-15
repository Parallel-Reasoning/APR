"""
Script for generating search paths for countdown problems using SoS+.
"""
import json
import argparse
import random
import os, sys
import multiprocessing as mp
from functools import partial
import tqdm

from src.countdown_utils import sum_heuristic, mult_heuristic, simple_rating
from src.data.search import sosp

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--problems_path", type=str, required=True, help="Path to problems JSON file")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to store search results")
parser.add_argument("--num_workers", type=int, default=int(mp.cpu_count()/4), help="Number of worker processes")
parser.add_argument("--max_beam_size", type=int, default=5, 
                    help="Maximum beam size for random selection")
def generate_search(args, problem, _=None):
    target = problem["target"]
    nums = problem["nums"]
    start_size = problem["start_size"]
    
    if start_size == 2:
        max_rating = 4
    elif start_size == 3:
        max_rating = 3*4*4
    elif start_size == 4:
        max_rating = 1152
    elif start_size == 5:
        max_rating = 46080
    elif start_size == 6:
        max_rating = 2764800

    heuristic = random.choice([sum_heuristic, mult_heuristic])
    beam_size = random.choice(range(1, args.max_beam_size + 1))
    search_path = sosp(target, nums, beam_size, heuristic=mult_heuristic)

    if "Goal Reached" in search_path:
        rating = 1. - simple_rating(search_path) / max_rating
        rating = max(0., rating)
    else:
        rating = 0.

    search_type = f"sosp_{beam_size}"
    result = {
        **problem,  # Include original problem data
        "search_path": search_path,
        "rating": rating,
        "search_type": search_type,
        "heuristic": heuristic.__name__
    }
        
    return result

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Using {args.num_workers} CPU workers")
    random.seed(args.seed)

    # Load problems
    with open(args.problems_path) as f:
        problems = json.load(f)

    # Generate searches in parallel
    pool = mp.Pool(args.num_workers)
    gen_search = partial(generate_search, args)
    results = list(tqdm.tqdm(pool.imap(gen_search, problems), total=len(problems)))
    
    pool.close()
    pool.join()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{os.path.basename(args.problems_path).replace('_problems.json', '')}_sosp.json")
    print(output_path)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4) 