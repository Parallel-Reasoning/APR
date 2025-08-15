"""
Script for generating search paths for countdown problems using APR.
"""
import json
import argparse
import random
import os, sys
import multiprocessing as mp
from functools import partial
import tqdm
import numpy as np
from transformers import AutoTokenizer

from src.countdown_utils import sum_heuristic, mult_heuristic, simple_rating
from src.data.search import apr

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--problems_path", type=str, 
                    help="Path to problems JSON file")
parser.add_argument("--output_dir", type=str, default="data", help="Directory to store search results")
parser.add_argument("--num_workers", type=int, default=int(mp.cpu_count()/2), help="Number of worker processes")
parser.add_argument("--total_samples", type=int, default=10000, help="Total number of samples to generate")
parser.add_argument("--min_beam_size", type=int, default=None, 
                    help="Minimum beam size for random selection")
parser.add_argument("--max_beam_size", type=int, default=6, 
                    help="Maximum beam size for random selection")
parser.add_argument("--max_sub_call_beam_size", type=int, default=10, 
                    help="Maximum sub call beam size for random selection")
parser.add_argument("--promising_threshold", type=float, default=0.9, 
                    help="Promising threshold for hybrid search")

# Global tokenizer for workers
_tokenizer = None

def init_worker():
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def get_token_length(text):
    """Calculate the number of tokens in a string using the proper method."""
    global _tokenizer
    tokens = _tokenizer.encode(_tokenizer.bos_token + text + _tokenizer.eos_token)
    return len(tokens)

def add_all_calls(trace_dict):
    """Recursively collect all call traces from a trace_dict."""
    all_calls = []
    all_calls += trace_dict['main_calls']
    for sub in trace_dict['sub_calls']:
        for sub_trace in sub:
            all_calls += add_all_calls(sub_trace)
    return all_calls

def calculate_token_counts(trace_dict, tokenizer):
    """Calculate token counts for the trace_dict including longest and average sequence tokens."""
    seqs = add_all_calls(trace_dict)
    
    # Initialize token count data
    token_count = 0
    longest_seq_token_count = 0
    avg_seq_token_count = 0
    avg_sub_seq_token_count = 0
    sub_call_count = 0
    
    if len(seqs) > 1:
        # Find all sequences that start with "Moving to Node #0"
        root_seqs = [seq for seq in seqs if seq.startswith("Moving to Node #0\n")]
        sub_seqs = [seq for seq in seqs if not seq.startswith("Moving to Node #0\n")]
        
        # Count sub calls
        sub_call_count = len(sub_seqs)
        sub_call_count /= (len(root_seqs) - 1)

        # assert sub_call_count <= 5, f"Sub call count is {sub_call_count}"
        
        # Sort root sequences by length (shortest first)
        root_seqs = sorted(root_seqs, key=len)
        
        # Calculate total tokens without considering redundancy
        total_tokens = 0
        sub_seq_tokens = 0
        # Calculate total token count and track longest sequence in one pass
        for seq in seqs:
            tokens = tokenizer.encode(tokenizer.bos_token + seq + tokenizer.eos_token)
            total_tokens += len(tokens)
            longest_seq_token_count = max(longest_seq_token_count, len(tokens))
        
        # Calculate sub sequence token counts
        if sub_seqs:
            for seq in sub_seqs:
                tokens = tokenizer.encode(tokenizer.bos_token + seq + tokenizer.eos_token)
                sub_seq_tokens += len(tokens)
            avg_sub_seq_token_count = sub_seq_tokens / len(sub_seqs)
        
        # Calculate redundant tokens between root sequences
        redundant_tokens = 0
        if len(root_seqs) > 1:
            # Find common prefixes between each pair of sequences
            for i in range(len(root_seqs) - 1):
                j = i + 1
                seq1 = root_seqs[i]
                seq2 = root_seqs[j]
                
                # Find common prefix
                prefix_len = 0
                for k in range(min(len(seq1), len(seq2))):
                    if seq1[k] == seq2[k]:
                        prefix_len += 1
                    else:
                        break
                
                if prefix_len > 0:
                    common_prefix = seq1[:prefix_len]
                    # Count tokens in this prefix
                    prefix_tokens = len(tokenizer.encode(common_prefix)) - 2  # Subtract BOS/EOS
                    redundant_tokens += max(0, prefix_tokens)
        
        # Final token count is total minus redundant
        token_count = total_tokens - redundant_tokens
        avg_seq_token_count = token_count / len(seqs)
    else:
        tokens = tokenizer.encode(tokenizer.bos_token + seqs[0] + tokenizer.eos_token)
        token_count = len(tokens)
        longest_seq_token_count = token_count
        avg_seq_token_count = token_count
        sub_call_count = 0
    
    return {
        "token_count": token_count,
        "longest_seq_token_count": longest_seq_token_count,
        "avg_seq_token_count": avg_seq_token_count,
        "avg_sub_seq_token_count": avg_sub_seq_token_count,
        "sub_call_count": sub_call_count
    }

def generate_search(problem, args):
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
    
    # Random beam size between 1 and max_beam_size
    if args.min_beam_size:
        beam_size = random.randint(args.min_beam_size, args.max_beam_size)
    else:
        beam_size = random.randint(1, args.max_beam_size)
    
    # Random sub_call_beam_size between beam_size and max_sub_call_beam_size
    if args.max_sub_call_beam_size > beam_size: 
        sub_call_beam_size = random.randint(beam_size, args.max_sub_call_beam_size)
        # sub_call_beam_size = args.max_sub_call_beam_size
    else:
        sub_call_beam_size = args.max_sub_call_beam_size
    
    # Use hybrid search with the selected parameters
    heuristic = mult_heuristic  # Using mult_heuristic as in the original code
    promising_threshold = args.promising_threshold  # Fixed value
    trace_dict = apr(target, nums, beam_size, sub_call_beam_size=sub_call_beam_size, 
                    promising_threshold=promising_threshold, heuristic=heuristic)
    search_path = trace_dict["main_calls"][-1]
    
    # Calculate token counts
    token_data = calculate_token_counts(trace_dict, _tokenizer)
    
    result = {
        **problem,  # Include original problem data
        "search_path": search_path,
        "rating": 1.0 - simple_rating(search_path) / max_rating if "Goal Reached" in search_path else 0.0,
        "search_type": f"apr_{beam_size}",
        "heuristic": heuristic.__name__,
        "promising_threshold": promising_threshold,
        "beam_size": beam_size,
        "sub_call_beam_size": sub_call_beam_size,
        "trace_dict": trace_dict,
        **token_data,  # Add token count data
    }
    
    return result

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Using {args.num_workers} CPU workers")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load problems
    with open(args.problems_path) as f:
        problems = json.load(f)
    
    # Take only the number of problems needed
    problems = problems[:args.total_samples]
    
    print(f"Using {len(problems)} problems from {args.problems_path}")
    print(f"Using fixed promising_threshold: {args.promising_threshold}")
    print(f"Using max_beam_size: {args.max_beam_size}, max_sub_call_beam_size: {args.max_sub_call_beam_size}")

    # Generate searches in parallel, initializing tokenizer in each worker
    pool = mp.Pool(args.num_workers, initializer=init_worker)
    gen_func = partial(generate_search, args=args)
    
    # Use chunksize to avoid overhead
    results = list(tqdm.tqdm(
        pool.imap(gen_func, problems, chunksize=10),
        total=len(problems),
        desc="Generating searches"
    ))
    
    pool.close()
    pool.join()
    
    # Count successful solutions
    successful = [r for r in results if r["rating"] > 0]
    acc = len(successful) / len(problems) if problems else 0
    print(f"Accuracy: {acc:.4f} ({len(successful)} successful solutions out of {len(problems)} problems)")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    prom_threshold = str(args.promising_threshold).replace(".", "_")
    output_filename = f"train_apr.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print total token statistics
    total_token_counts = [r["token_count"] for r in results]
    if total_token_counts:
        avg_total_tokens = sum(total_token_counts) / len(total_token_counts)
        print(f"Total token statistics: Avg={avg_total_tokens:.2f}")

    # Print longest sequence token statistics
    token_counts = [r["longest_seq_token_count"] for r in results]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        print(f"Longest sequence token statistics: Avg={avg_tokens:.2f}, Max={max_tokens}, Min={min_tokens}")
    
    # Print sub sequence token statistics
    sub_seq_token_counts = [r["avg_sub_seq_token_count"] for r in results]
    sub_seq_token_counts = [count for count in sub_seq_token_counts if count > 0]
    if sub_seq_token_counts:
        avg_sub_seq_tokens = sum(sub_seq_token_counts) / len(sub_seq_token_counts)
        max_sub_seq_tokens = max(sub_seq_token_counts)
        min_sub_seq_tokens = min(sub_seq_token_counts)
        print(f"Sub sequence token statistics: Avg={avg_sub_seq_tokens:.2f}, Max={max_sub_seq_tokens:.2f}, Min={min_sub_seq_tokens:.2f}")
    
    # Print sub call statistics
    sub_call_counts = [r["sub_call_count"] for r in results]
    if sub_call_counts:
        avg_sub_calls = sum(sub_call_counts) / len(sub_call_counts)
        max_sub_calls = max(sub_call_counts)
        min_sub_calls = min(sub_call_counts)
        print(f"Sub call statistics: Avg={avg_sub_calls:.2f}, Max={max_sub_calls}, Min={min_sub_calls}")
    
    # Print beam size and sub call beam size statistics
    beam_sizes = [r["beam_size"] for r in results]
    sub_call_beam_sizes = [r["sub_call_beam_size"] for r in results]
    
    if beam_sizes:
        avg_beam_size = sum(beam_sizes) / len(beam_sizes)
        print(f"Beam size statistics: Avg={avg_beam_size:.2f}")
    
    if sub_call_beam_sizes:
        avg_sub_beam_size = sum(sub_call_beam_sizes) / len(sub_call_beam_sizes)
        print(f"Sub call beam size statistics: Avg={avg_sub_beam_size:.2f}") 