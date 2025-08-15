import heapq
import itertools
import random

from src.countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn, get_path_to_target
from copy import deepcopy


def is_promising(promising_threshold=0.9):
    # return False
    return random.random() < promising_threshold


def _search(target, start_node, beam_size, threshold=None, promising_threshold=0.5, heuristic=sum_heuristic, level=0, root_node=None):
    search_trace = ""
    # open_set is a priority queue we use to store the nodes to explore
    open_set = []
    # Push the initial node with its index, heuristic value, and parent index
    heapq.heappush(open_set, (heuristic(start_node.nums, target), start_node))

    while open_set:
        # we explore beam_size nodes at a time, prirotized by ones with the best heuristic value. It's not actually layer by layer
        # layer 0 has 1 node
        # layer 1 has beam_size nodes
        # layer N has beam_size^N nodes
        # where are we doing the pruning? 
        # we only generate beam_size successors for each node and drop the rest

        # Get the top beam width nodes based on heuristic (pruning others)
        current_nodes = [heapq.heappop(open_set) for _ in range(min(beam_size, len(open_set)))]
        if not current_nodes:
            break  # Exit if no nodes are left to expand
        for idx, (_, current_node) in enumerate(current_nodes):
            search_trace += f"Current State: {target}:{current_node.nums}, Operations: {current_node.operations}\n"
            # Store nodes generated at this level for pruning
            generated_nodes = []
            # Generate successors for each node
            for i, j in itertools.combinations(range(len(current_node.nums)), 2):
                for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                    new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                    new_operations = current_node.operations + [operation]
                    new_heuristic = heuristic(new_nums, target)
                    new_node = CountdownNode(None, current_node, new_nums, new_operations, new_heuristic)
                    generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes

            # Explicit Pruning: Keep only top 'beam_size' nodes, prune the rest
            if threshold is None:
                # if beam_size bfs-like pruning
                generated_nodes.sort()  # Sort by heuristic value
                pruned_nodes = generated_nodes[beam_size:]  # Nodes that will be pruned
                generated_nodes = generated_nodes[:beam_size]  # Nodes that will be kept
            else:
                # if threshold dfs-like pruning
                generated_nodes = [node for node in generated_nodes if node[0] <= threshold]

           # shuffle the generated nodes so that the first node explored is not the same every time
            random.shuffle(generated_nodes)
            # Add remaining nodes back to open_set
            node_idx = 0
            nodes_to_explore = []
            for node_tuple in generated_nodes:
                node = node_tuple[-1]
                node.idx = f"{node.parent.idx},{node_idx}"
                operation = node.operations[-1]
                nums = node.nums
                search_trace += f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
                if len(node.nums) == 1 and node.nums[0] == target:
                    search_trace += f"{node.nums[0]},{target} equal: Goal Reached\n"
                    search_trace += get_path_to_target(node, root_node)
                    return search_trace
                elif len(new_node.nums) == 1:
                    search_trace += f"{node.nums[0]},{target} unequal: No Solution\n"
                else:
                    # here we are adding the node to the open set
                    search_trace += f"Generated Node #{node.idx}: {target}:{node.nums} Operation: {operation}\n"
                    node_idx += 1
                    nodes_to_explore.append(node)
            nodes_to_explore.sort()
            double_down = is_promising(promising_threshold)
            for node in nodes_to_explore:
                if double_down:
                    # is promosing, we double down by search on this
                    sub_search_trace = f"Start Sub Search at level {level+1}: Moving to Node #{node.idx}\n"
                    sub_search_trace += _search(target, deepcopy(node), beam_size=beam_size, threshold=threshold, promising_threshold=promising_threshold, heuristic=heuristic, level=level+1, root_node=root_node)
                    sub_search_trace += f"Exit Sub Search at level {level+1}\n"
                    # indented by = for each line
                    # sub_search_trace = "\n".join([f"={line}" for line in sub_search_trace.split("\n")])
                    search_trace += sub_search_trace
                    if "Goal Reached" in sub_search_trace:
                        return search_trace
                    else:
                        search_trace += "No solution found in sub search\n"
                else:
                    heapq.heappush(open_set, (node.heuristic, node))
 
            # Note transition to the next node within the current set
            if idx < len(current_nodes) - 1:
                _, next_node = current_nodes[idx + 1]
                next_index = next_node.idx
                search_trace += f"Moving to Node #{next_index}\n"
            

        # Backtracking trace
        if current_nodes:
            # Find the next node to backtrack to (the next in the open set)
            next_node_to_explore = open_set[0] if open_set else None
            if next_node_to_explore:
                _, next_node = next_node_to_explore
                next_index = next_node.idx
                search_trace += f"Moving to Node #{next_index}\n"

    search_trace += "No solution found.\n"
    return search_trace

def sosp(target, nums, beam_size, threshold=None, promising_threshold=0.5, heuristic=sum_heuristic):
    # beam_size is always used for parallel search
    # but for pruning, if threshold is None, it's bfs-like pruning by beam size,
    # if threshold is not None, it's dfs-like pruning by threshold
    start_node = CountdownNode(0, None, nums, [], heuristic(nums, target))
    return _search(target, start_node, beam_size, threshold=threshold, promising_threshold=promising_threshold, heuristic=heuristic, root_node=start_node)


def _parallel_search(
    target, start_node, beam_size, sub_call_beam_size, threshold=None, promising_threshold=0.9, heuristic=sum_heuristic, level=0):
    # trace_dict tracks the search process across different calls:
    # - main_calls: List of strings, each element represents a main call's trace
    #   - main_calls[0]: Initial call trace
    #   - main_calls[1]: Trace after first sub-process calls
    #   - main_calls[i]: Trace after i-th sub-process calls
    # - sub_calls: List of traces from recursive calls made during sub-problem solving. They are also trace_dicts
    #   - sub_calls[0]: a list of trace_dicts from the first sub-process call
    #   - sub_calls[i]: a list of trace_dicts from the i-th sub-process call
    trace_dict = {
        "main_calls": [""],  # Initialize with empty string for first main call
        "sub_calls": [],     # Will store traces from recursive sub-problem calls
    }
    # open_set is a priority queue we use to store the nodes to explore
    open_set = []
    # Push the initial node with its index, heuristic value, and parent index
    heapq.heappush(open_set, (heuristic(start_node.nums, target), start_node))
    trace_dict["main_calls"][-1] += f"Moving to Node #{start_node.idx}\n"

    while open_set:
        # we explore beam_size nodes at a time, prirotized by ones with the best heuristic value. It's not actually layer by layer
        # layer 0 has 1 node
        # layer 1 has beam_size nodes
        # layer N has beam_size^N nodes
        # where are we doing the pruning? 
        # we only generate beam_size successors for each node and drop the rest

        # Get the top beam width nodes based on heuristic (pruning others)
        current_nodes = [heapq.heappop(open_set) for _ in range(min(beam_size, len(open_set)))]
        if not current_nodes:
            break  # Exit if no nodes are left to expand
        for idx, (_, current_node) in enumerate(current_nodes):
            trace_dict["main_calls"][-1] += f"Current State: {target}:{current_node.nums}, Operations: {current_node.operations}\n"
            # Store nodes generated at this level for pruning
            generated_nodes = []
            # Generate successors for each node
            for i, j in itertools.combinations(range(len(current_node.nums)), 2):
                for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                    new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                    new_operations = current_node.operations + [operation]
                    new_heuristic = heuristic(new_nums, target)
                    new_node = CountdownNode(None, current_node, new_nums, new_operations, new_heuristic)
                    generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes

            # Explicit Pruning: Keep only top 'beam_size' nodes, prune the rest
            if threshold is None:
                # if beam_size bfs-like pruning
                generated_nodes.sort()  # Sort by heuristic value
                pruned_nodes = generated_nodes[beam_size:]  # Nodes that will be pruned
                generated_nodes = generated_nodes[:beam_size]  # Nodes that will be kept
            else:
                raise NotImplementedError("Threshold pruning is not implemented yet")
                # if threshold dfs-like pruning
                generated_nodes = [node for node in generated_nodes if node[0] <= threshold]

           # shuffle the generated nodes so that the first node explored is not the same every time
            random.shuffle(generated_nodes)
            # Add remaining nodes back to open_set
            node_idx = 0
            nodes_to_explore = []
            for node_tuple in generated_nodes:
                node = node_tuple[-1]
                node.idx = f"{node.parent.idx},{node_idx}"
                operation = node.operations[-1]
                nums = node.nums
                trace_dict["main_calls"][-1] += f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
                if len(node.nums) == 1 and node.nums[0] == target:
                    trace_dict["main_calls"][-1] += f"{node.nums[0]},{target} equal: Goal Reached\n"
                    trace_dict["main_calls"][-1] += get_path_to_target(node, start_node)
                    return trace_dict, node
                elif len(new_node.nums) == 1:
                    trace_dict["main_calls"][-1] += f"{node.nums[0]},{target} unequal: No Solution\n"
                else:
                    # here we are adding the node to the open set
                    trace_dict["main_calls"][-1] += f"Generated Node #{node.idx}: {target}:{node.nums} Operation: {operation}\n"
                    node_idx += 1
                    nodes_to_explore.append(node)
            nodes_to_explore.sort()
            if level == 0 and len(nodes_to_explore) > 1:
                # TODO: we do not nest sub searches yet
                double_down = is_promising(promising_threshold)
            else:
                double_down = False
            if double_down:
                # is promising, we double down by searching all nodes at this level
                # this will add sub_calls to traces and we will also start a new main call
                sub_calls = []
                trace_dict["sub_calls"].append([])
                for idx, node in enumerate(nodes_to_explore):
                    sub_trace_dict, sub_goal_node = _parallel_search(target, deepcopy(node), beam_size=sub_call_beam_size, 
                                                        sub_call_beam_size=sub_call_beam_size,
                                                        threshold=threshold, promising_threshold=promising_threshold, 
                                                        heuristic=heuristic, level=level+1)
                    sub_calls.append((sub_trace_dict, sub_goal_node))
                    trace_dict["sub_calls"][-1].append(sub_trace_dict)
                # Finish the main call that started this sub search
                main_trace_before_sub_calls = deepcopy(trace_dict["main_calls"][-1])
                trace_dict["main_calls"][-1] += "<Calling Sub Searches>\n"
                for idx, node in enumerate(nodes_to_explore):
                    trace_dict["main_calls"][-1] += f"<Start Sub Search {idx} at level {level+1}> Moving to Node #{node.idx}\n"
                trace_dict["main_calls"][-1] += "<End Calling Sub Searches>\n"
                # ---  Instantiate the new main call ---
                trace_dict["main_calls"].append(main_trace_before_sub_calls)
                trace_dict["main_calls"][-1] += "<Sub Searches>\n" 
                for idx, (sub_trace_dict, sub_goal_node) in enumerate(sub_calls):
                    if sub_goal_node:
                        # So goal is found in sub search
                        trace_dict["main_calls"][-1] += f"<Goal Reached in Sub Search {idx} at level {level+1} at Node #{nodes_to_explore[idx].idx}>\n"
                        trace_dict["main_calls"][-1] += get_path_to_target(sub_goal_node, nodes_to_explore[idx])
                        # This is the call back from the sub search
                    else:
                        # No goal is found in sub search
                        trace_dict["main_calls"][-1] += f"<No Solution in Sub Search {idx} at level {level+1} at Node #{nodes_to_explore[idx].idx}>\n"
                # We check again if any sub search found a solution
                trace_dict["main_calls"][-1] += "<End Sub Searches>\n"
                for idx, (sub_trace_dict, sub_goal_node) in enumerate(sub_calls):
                    if sub_goal_node:
                        # So goal is found in sub search
                        # 
                        trace_dict["main_calls"][-1] += f"{sub_goal_node.nums[0]},{target} equal: Goal Reached\n"
                        trace_dict["main_calls"][-1] += get_path_to_target(sub_goal_node, start_node)
                        return trace_dict, sub_goal_node
                
            else:
                # add all nodes to open set for breadth-first exploration
                for node in nodes_to_explore:
                    heapq.heappush(open_set, (node.heuristic, node))
 
            # Note transition to the next node within the current set
            if idx < len(current_nodes) - 1:
                _, next_node = current_nodes[idx + 1]
                next_index = next_node.idx
                trace_dict["main_calls"][-1] += f"Moving to Node #{next_index}\n"
            

        # Backtracking trace
        if current_nodes:
            # Find the next node to backtrack to (the next in the open set)
            next_node_to_explore = open_set[0] if open_set else None
            if next_node_to_explore:
                _, next_node = next_node_to_explore
                next_index = next_node.idx
                trace_dict["main_calls"][-1] += f"Moving to Node #{next_index}\n"

    trace_dict["main_calls"][-1] += "No solution found.\n"
    return trace_dict, None

def apr(target, nums, beam_size, sub_call_beam_size=None, threshold=None, promising_threshold=0.9, heuristic=sum_heuristic):
    # beam_size is always used for parallel search
    # but for pruning, if threshold is None, it's bfs-like pruning by beam size,
    # if threshold is not None, it's dfs-like pruning by threshold
    start_node = CountdownNode(0, None, nums, [], heuristic(nums, target))
    if sub_call_beam_size is None:
        sub_call_beam_size = beam_size
    return _parallel_search(target, start_node, beam_size, sub_call_beam_size, threshold=threshold, promising_threshold=promising_threshold, heuristic=heuristic)[0]


if __name__ == "__main__":
    # Example usage
    random.seed(4)
    target = 24
    nums = [4,9,3]
    search_path = apr(target, nums, beam_size=3, promising_threshold=0.9, heuristic=mult_heuristic)
    print(search_path)
    print(len(search_path))
    print(metric_fn(search_path))