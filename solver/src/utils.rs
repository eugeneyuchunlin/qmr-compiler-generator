use crate::structures::*;

use itertools::{max, Itertools};
use petgraph::graph::{Node, NodeIndex};
use petgraph::Direction::Outgoing;
use petgraph::Graph;
use rand::seq::IndexedRandom;
use rand::Rng;
use regex::Regex;
use rustworkx_core::steiner_tree::steiner_tree;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use std::iter::from_fn;

#[derive(Debug)]
pub enum IOError {
    InputErr,
    OutputErr(serde_json::Error),
}

pub fn extract_cnots(filename: &str) -> Circuit {
    let file = File::open(filename).unwrap();
    let lines = io::BufReader::new(file).lines();
    let mut gates = Vec::new();
    let mut qubits = HashSet::new();
    let mut id = 0;
    let cx_re = Regex::new(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\];").unwrap();
    for line in lines {
        let line_str = line.unwrap();
        let cx_caps = cx_re.captures(&line_str);
        match cx_caps {
            None => continue,
            Some(c) => {
                let q1 = Qubit::new(c.get(1).unwrap().as_str().parse::<usize>().unwrap());
                let q2 = Qubit::new(c.get(2).unwrap().as_str().parse::<usize>().unwrap());
                qubits.insert(q1);
                qubits.insert(q2);
                let gate = Gate {
                    operation: Operation::CX,
                    qubits: vec![q1, q2],
                    id,
                };
                gates.push(gate);
                id += 1;
            }
        }
    }
    return Circuit { gates, qubits };
}

pub fn extract_scmr_gates(filename: &str) -> Circuit {
    let file = File::open(filename).unwrap();
    let lines = io::BufReader::new(file).lines();
    let mut gates = Vec::new();
    let mut qubits = HashSet::new();
    let mut id = 0;
    let cx_re = Regex::new(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\];").unwrap();
    let t_re = Regex::new(r"(t|tdg)\s+q\[(\d+)\];").unwrap();
    for line in lines {
        let line_str = line.unwrap();
        let cx_caps = cx_re.captures(&line_str);
        let t_caps = t_re.captures(&line_str);
        match cx_caps {
            None => match t_caps {
                None => continue,
                Some(c) => {
                    let q = Qubit::new(c.get(2).unwrap().as_str().parse::<usize>().unwrap());
                    qubits.insert(q);
                    let gate = Gate {
                        operation: Operation::T,
                        qubits: vec![q],
                        id,
                    };
                    gates.push(gate);
                    id += 1;
                }
            },
            Some(c) => {
                let q1 = Qubit::new(c.get(1).unwrap().as_str().parse::<usize>().unwrap());
                let q2 = Qubit::new(c.get(2).unwrap().as_str().parse::<usize>().unwrap());
                qubits.insert(q1);
                qubits.insert(q2);
                let gate = Gate {
                    operation: Operation::CX,
                    qubits: vec![q1, q2],
                    id,
                };
                gates.push(gate);
                id += 1;
            }
        }
    }
    return Circuit { gates, qubits };
}

fn parse_pauli_term(c: char) -> PauliTerm {
    match c {
        'I' => PauliTerm::PauliI,
        'X' => PauliTerm::PauliX,
        'Y' => PauliTerm::PauliY,
        'Z' => PauliTerm::PauliZ,
        _ => panic!("Invalid Pauli term"),
    }
}
type GateHandler = Box<dyn FnMut(&regex::Captures, &mut HashSet<Qubit>, usize) -> Gate>;

pub fn extract_gates(filename: &str, gate_types: &[&str]) -> Circuit {
    let file = File::open(filename).unwrap();
    let lines = io::BufReader::new(file).lines();
    let mut gates = Vec::new();
    let mut qubits = HashSet::new();
    let mut id = 0;
    let mut patterns: Vec<(Regex, GateHandler)> = vec![];
    if gate_types.contains(&"CX") {
        let cx_pattern = (
            Regex::new(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\];").unwrap(),
            Box::new(|c: &regex::Captures, qubits: &mut HashSet<Qubit>, id| {
                let q1 = Qubit::new(c.get(1).unwrap().as_str().parse::<usize>().unwrap());
                let q2 = Qubit::new(c.get(2).unwrap().as_str().parse::<usize>().unwrap());
                qubits.insert(q1);
                qubits.insert(q2);
                Gate {
                    operation: Operation::CX,
                    qubits: vec![q1, q2],
                    id,
                }
            }) as GateHandler,
        );
        patterns.push(cx_pattern);
    }
    if gate_types.contains(&"T") {
        let t_pattern = (
            Regex::new(r"(t|tdg)\s+q\[(\d+)\];").unwrap(),
            Box::new(
                |c: &regex::Captures, qubits: &mut HashSet<Qubit>, id: usize| {
                    let q = Qubit::new(c.get(2).unwrap().as_str().parse::<usize>().unwrap());
                    qubits.insert(q);
                    Gate {
                        operation: Operation::T,
                        qubits: vec![q],
                        id,
                    }
                },
            ) as GateHandler,
        );

        patterns.push(t_pattern);
    }

    if gate_types.contains(&"Pauli") {
        let paul_rot_pattern = (
            Regex::new(r"([IXYZ]+)_\((-?\d+)/(\d+)\);").unwrap(),
            Box::new(
                |c: &regex::Captures, qubits: &mut HashSet<Qubit>, id: usize| {
                    let axis_str = c.get(1).unwrap().as_str();
                    let numerator = c.get(2).unwrap().as_str().parse::<isize>().unwrap();
                    let denominator = c.get(3).unwrap().as_str().parse::<usize>().unwrap();
                    let axis: Vec<PauliTerm> = axis_str.chars().map(parse_pauli_term).collect();
                    let nontrivial_indices =
                        (0..axis.len()).filter(|ind| axis[*ind] != PauliTerm::PauliI);
                    let gate_qubits: Vec<Qubit> = nontrivial_indices.map(Qubit::new).collect();
                    qubits.extend(gate_qubits.iter());
                    Gate {
                        operation: Operation::PauliRot {
                            axis,
                            angle: (numerator, denominator),
                        },
                        qubits: gate_qubits,
                        id,
                    }
                },
            ) as GateHandler,
        );
        let paul_meas_pattern = (
            Regex::new(r"(-?)M_([IXYZ]+);").unwrap(),
            Box::new(
                |c: &regex::Captures, qubits: &mut HashSet<Qubit>, id: usize| {
                    let sign_str = c.get(1).unwrap().as_str();
                    let sign = sign_str != "-";
                    let axis_str = c.get(2).unwrap().as_str();
                    let axis: Vec<PauliTerm> = axis_str.chars().map(parse_pauli_term).collect();
                    let nontrivial_indices =
                        (0..axis.len()).filter(|ind| axis[*ind] != PauliTerm::PauliI);
                    let gate_qubits: Vec<Qubit> = nontrivial_indices.map(Qubit::new).collect();
                    qubits.extend(gate_qubits.iter());
                    Gate {
                        operation: Operation::PauliMeasurement { sign, axis },
                        qubits: gate_qubits,
                        id,
                    }
                },
            ) as GateHandler,
        );
        patterns.push(paul_rot_pattern);
        patterns.push(paul_meas_pattern);
    }
    for line in lines {
        let line_str = line.unwrap();
        for (regex, handler) in &mut patterns {
            if let Some(caps) = regex.captures(&line_str) {
                let gate = handler(&caps, &mut qubits, id);
                gates.push(gate);
                id += 1;
            }
        }
    }

    return Circuit { gates, qubits };
}

pub fn path_graph(n: usize) -> Graph<Location, ()> {
    let mut g = Graph::new();
    let mut nodes = Vec::new();
    for i in 0..n {
        nodes.push(g.add_node(Location::new(i)));
    }
    for i in 0..n - 1 {
        g.add_edge(nodes[i], nodes[i + 1], ());
        g.add_edge(nodes[i + 1], nodes[i], ());
    }
    return g;
}

pub fn drop_zeros_and_normalize<T: IntoIterator<Item = (f64, f64)> + Clone>(
    weighted_values: T,
) -> f64 {
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;
    for (w, v) in weighted_values.clone() {
        if v != 0.0 {
            total_weight += w;
        }
    }
    for (w, v) in weighted_values.clone() {
        {
            let normalized = w / total_weight;
            weighted_sum += normalized * v;
        }
    }
    return weighted_sum;
}

fn graph_from_edge_vec(edges: Vec<(Location, Location)>) -> Graph<Location, ()> {
    let mut nodes = HashMap::new();
    let mut g = Graph::new();
    for (a, b) in &edges {
        if !nodes.contains_key(a) {
            nodes.insert(a, g.add_node(*a));
        }
        if !nodes.contains_key(b) {
            nodes.insert(b, g.add_node(*b));
        }
        // edges are undirected
        g.update_edge(nodes[a], nodes[b], ());
        g.update_edge(nodes[b], nodes[a], ());
    }
    return g;
}

pub fn graph_from_file(filename: &str) -> Graph<Location, ()> {
    let file = File::open(filename).unwrap();
    let parsed: Value = serde_json::from_reader(file).unwrap();
    let edges = parsed
        .as_array()
        .expect("Expected an array of arrays")
        .iter()
        .map(|inner| {
            let array = inner.as_array().expect("Inner element is not an array");
            if array.len() != 2 {
                panic!("Each edge must have exactly 2 elements");
            }
            let first = array[0]
                .as_u64()
                .expect("Element is not a positive integer") as usize;
            let second = array[1]
                .as_u64()
                .expect("Element is not a positive integer") as usize;
            (Location::new(first), Location::new(second))
        })
        .collect();
    return graph_from_edge_vec(edges);
}

pub fn graph_from_json_entry(entry: Value) -> Graph<Location, ()> {
    let edges = entry
        .as_array()
        .expect("Expected an array of arrays")
        .iter()
        .map(|inner| {
            let array = inner.as_array().expect("Inner element is not an array");
            if array.len() != 2 {
                panic!("Each edge must have exactly 2 elements");
            }
            let first = array[0]
                .as_u64()
                .expect("Element is not a positive integer") as usize;
            let second = array[1]
                .as_u64()
                .expect("Element is not a positive integer") as usize;
            (Location::new(first), Location::new(second))
        })
        .collect();
    return graph_from_edge_vec(edges);
}

pub fn vertical_neighbors(loc: Location, width: usize, height: usize) -> Vec<Location> {
    let mut neighbors = Vec::new();
    if loc.get_index() / width > 0 {
        neighbors.push(Location::new(loc.get_index() - width));
    }
    if loc.get_index() / width < height - 1 {
        neighbors.push(Location::new(loc.get_index() + width));
    }
    return neighbors;
}

pub fn multiple_horizontal_neighbors(locs: Vec<Location>, width: usize) -> Vec<Location> {
    let mut neighbors = Vec::new();
    for loc in locs {
        neighbors.extend(horizontal_neighbors(loc, width));
    }
    return neighbors; 
}

pub fn horizontal_neighbors(loc: Location, width: usize) -> Vec<Location> {
    let mut neighbors = Vec::new();
    if loc.get_index() % width > 0 {
        neighbors.push(Location::new(loc.get_index() - 1));
    }
    if loc.get_index() % width < width - 1 {
        neighbors.push(Location::new(loc.get_index() + 1));
    }
    return neighbors;
}

pub fn swap_keys(
    map: &HashMap<Qubit, Location>,
    loc1: Location,
    loc2: Location,
) -> HashMap<Qubit, Location> {
    let mut new_map = map.clone();
    for (qubit, loc) in map {
        if loc == &loc1 {
            new_map.insert(*qubit, loc2);
        } else if loc == &loc2 {
            new_map.insert(*qubit, loc1);
        }
    }
    return new_map;
}

pub fn push_and_return<T: Clone, C: Clone + IntoIterator<Item = T>>(coll: C, item: T) -> Vec<T> {
    let mut new: Vec<T> = coll.into_iter().collect();
    new.push(item);
    return new;
}

pub fn extend_and_return<
    C: Clone + IntoIterator<Item = T>,
    D: Clone + IntoIterator<Item = T>,
    T: Clone,
>(
    left: C,
    right: D,
) -> Vec<T> {
    let mut new: Vec<_> = left.clone().into_iter().collect();
    new.extend(right);
    return new.into_iter().collect();
}
pub fn values<T: Clone, U: Clone>(map: &HashMap<T, U>) -> Vec<U> {
    map.values().cloned().collect()
}

pub fn shortest_path<A: Architecture>(
    arch: &A,
    starts: Vec<Location>,
    ends: Vec<Location>,
    blocked: Vec<Location>,
) -> Option<Vec<Location>> {
    let (mut graph, mut loc_to_node) = arch.graph();
    for loc in blocked.iter() {
        let old_last = graph[graph.node_indices().last().unwrap()];
        graph.remove_node(loc_to_node[loc]);
        loc_to_node.insert(old_last, loc_to_node[loc]);
        loc_to_node.remove(loc);
    }
    let mut best: Option<(i32, Vec<NodeIndex>)> = None;
    for start in &starts {
        for end in &ends {
            if loc_to_node.contains_key(start) && loc_to_node.contains_key(end) {
                let res = petgraph::algo::astar(
                    &graph,
                    loc_to_node[&start],
                    |finish| finish == loc_to_node[&end],
                    |_e| 1,
                    |_| 0,
                );
                if best.is_none()
                    || ((&res).is_some() && &res.as_ref().unwrap().0 < &best.as_ref().unwrap().0)
                {
                    best = res;
                }
            }
        }
    }
    match best {
        None => None,
        Some((_, path)) => Some(path.into_iter().map(|x| graph[x]).collect()),
    }
}

pub fn identity_application<T: GateImplementation>(step: &Step<T>) -> Step<T> {
    return Step {
        implemented_gates: HashSet::new(),
        map: step.map.clone(),
    };
}
pub fn all_paths<A: Architecture>(
    arch: &A,
    starts: Vec<Location>,
    ends: Vec<Location>,
    blocked: Vec<Location>,
) -> impl Iterator<Item = Vec<Location>> {
    let (mut graph, mut loc_to_node) = arch.graph();
    let max_length = graph.node_count();
    // println!("graph: {:?}", graph);
    // println!("blocked: {:?}", blocked);
    let unique_blocked: Vec<Location> = blocked.iter().cloned().unique().collect(); 
    // println!("unique blocked: {:?}", unique_blocked);
    for loc in unique_blocked.iter() {
        let old_last = graph[graph.node_indices().last().unwrap()];
        // println!("removing node: {:?}", loc);
        graph.remove_node(loc_to_node[loc]);
        loc_to_node.insert(old_last, loc_to_node[loc]);
        loc_to_node.remove(loc);
    }

    let unblocked_starts: Vec<_> = starts
        .iter()
        .filter(|x| loc_to_node.contains_key(x))
        .cloned()
        .collect();
    let unblocked_ends: Vec<_> = ends
        .iter()
        .filter(|x| loc_to_node.contains_key(x))
        .cloned()
        .collect();
    let mut start_counter = 0;
    let mut visited = Vec::new();
    let mut stack: Vec<std::vec::IntoIter<NodeIndex>> = Vec::new();
    if !unblocked_starts.is_empty() {
        let start_neighbors: Vec<_> = graph
            .neighbors(loc_to_node[&unblocked_starts[start_counter]])
            .collect();
        stack.push(start_neighbors.into_iter());
        visited.push(unblocked_starts[start_counter]);
    }
    from_fn(move || {
        let mut exhausted = start_counter >= unblocked_starts.len();
        while !exhausted {
            if let Some(children) = stack.last_mut() {
                if let Some(child) = children.next() {
                    let loc = graph[child];
                    if visited.len() < max_length {
                        if ends.contains(&loc) {
                            let path: Vec<Location> =
                                visited.iter().chain(Some(&loc)).cloned().collect();
                            let n = path.len();
                            if path[n - 1] == path[0] {
                                return vec![path[0]].into();
                            } else {
                                return Some(path);
                            }
                        } else if !visited.contains(&loc) {
                            visited.push(loc);
                            let neighbors: Vec<_> =
                                graph.neighbors_directed(child, Outgoing).collect();
                            let n = neighbors.into_iter();
                            stack.push(n);
                        }
                    } else {
                        if unblocked_ends.contains(&graph[child])
                            || children.any(|x| unblocked_ends.contains(&graph[x]))
                        {
                            let path = visited.iter().chain(Some(&loc)).cloned().collect();
                            return Some(path);
                        }
                        stack.pop();
                        visited.pop();
                    }
                } else {
                    stack.pop();
                    visited.pop();
                }
            } else {
                start_counter += 1;
                if start_counter < unblocked_starts.len() {
                    visited = vec![unblocked_starts[start_counter]];
                    let start_neighbors: Vec<_> = graph
                        .neighbors(loc_to_node[&unblocked_starts[start_counter]])
                        .collect();
                    stack.push(start_neighbors.into_iter());
                } else {
                    exhausted = true;
                }
            }
        }
        None
    })
}

// pub fn steiner_trees<A: Architecture>(
//     arch: &A,
//     terminals: Vec<Vec<Location>>,
//     blocked: Vec<Location>,
// ) -> impl IntoIterator<Item = Vec<Location>> {
//     let (mut graph, mut loc_to_node) = arch.graph();
//     for loc in blocked.iter() {
//         let old_last = graph[graph.node_indices().last().unwrap()];
//         graph.remove_node(loc_to_node[loc]);
//         loc_to_node.insert(old_last, loc_to_node[loc]);
//         loc_to_node.remove(loc);
//     }
//     let terminal_sets = terminals
//         .into_iter()
//         .multi_cartesian_product()
//         .filter(|v| v.iter().all(|l| loc_to_node.contains_key(l)));
//     let mut impls = vec![];
//     for terminal_set in terminal_sets {
//         let indices: Vec<NodeIndex> = terminal_set.into_iter().map(|x| loc_to_node[&x]).collect();
//         let steiner_tree_res = steiner_tree(&graph, &indices, |_| Ok::<f64, ()>(1.0));

//         if let Ok(Some(tree)) = steiner_tree_res {
//             let locations = tree
//                 .used_node_indices
//                 .into_iter()
//                 .map(|n| &graph[NodeIndex::new(n)])
//                 .cloned()
//                 .collect();
//             impls.push(locations);
//         }
//     }
//     return impls;
// }

pub struct SteinerTreesIter {
    graph: Graph<Location, ()>,
    loc_to_node: HashMap<Location, NodeIndex>,
    terminal_sets: Vec<Vec<Location>>,
}

impl Iterator for SteinerTreesIter {
    type Item = Vec<Location>;

    fn next(&mut self) -> Option<Self::Item> {
        for terminal_set in &self.terminal_sets {
            let indices: Vec<NodeIndex> = terminal_set
                .into_iter()
                .map(|x| self.loc_to_node[&x])
                .collect();

            let steiner_tree_res = steiner_tree(&self.graph, &indices, |_| Ok::<f64, ()>(1.0));

            if let Ok(Some(tree)) = steiner_tree_res {
                let locations = tree
                    .used_node_indices
                    .into_iter()
                    .map(|n| &self.graph[NodeIndex::new(n)])
                    .cloned()
                    .collect();
                return Some(locations);
            }
        }
        None
    }
}

pub fn steiner_trees<A: Architecture>(
    arch: &A,
    terminals: Vec<Vec<Location>>,
    blocked: Vec<Location>,
) -> SteinerTreesIter {
    let (mut graph, mut loc_to_node) = arch.graph();

    // Remove blocked locations
    for loc in blocked.iter() {
        if let Some(node_idx) = loc_to_node.get(loc) {
            let old_last = graph[graph.node_indices().last().unwrap()];
            graph.remove_node(*node_idx);
            loc_to_node.insert(old_last, *node_idx);
            loc_to_node.remove(loc);
        }
    }

    // Create lazy iterator for terminal combinations
    let terminal_sets = terminals
        .into_iter()
        .multi_cartesian_product()
        .filter(|v| v.iter().all(|l| loc_to_node.contains_key(l)))
        .collect();

    SteinerTreesIter {
        graph,
        loc_to_node,
        terminal_sets,
    }
}

pub fn build_criticality_table(c: &Circuit) -> HashMap<usize, usize> {
    let mut qubit_table: HashMap<usize, usize> = HashMap::new();
    let mut gate_table: HashMap<usize, usize> = HashMap::new();
    for gate in &c.gates {
        let d = max(c.qubits.iter().map(|x| qubit_table.get(&x.get_index())))
            .flatten()
            .copied()
            .unwrap_or_default();
        gate_table.insert(gate.id, d + 1);
        for q in &c.qubits {
            qubit_table.insert(q.get_index(), d + 1);
        }
    }
    gate_table
}

pub fn build_interaction_graph(c: &Circuit) -> Graph<Qubit, usize> {
    let mut nodes = HashMap::new();
    let mut g = Graph::new();
    for qubit in &c.qubits {
        nodes.insert(*qubit, g.add_node(*qubit));
    }
    for gate in &c.gates {
        match &gate.operation {
            Operation::CX => {
                let (ctrl, tar) = (gate.qubits[0], gate.qubits[1]);
                let (ctrl_loc, tar_loc) = (
                    nodes
                        .get(&ctrl)
                        .expect("fetching control index in interaction graph"),
                    nodes
                        .get(&tar)
                        .expect("fetching target index in interaction graph"),
                );
                g.update_edge(*ctrl_loc, *tar_loc, 0);
                g.update_edge(*tar_loc, *ctrl_loc, 0);
            }
            Operation::T => continue,
            Operation::PauliRot { axis, angle: _ }
            | Operation::PauliMeasurement { sign: _, axis } => {
                // Iterate through all pairs of indices where the axis isn't PauliI
                for (i, term_i) in axis.iter().enumerate() {
                    if *term_i == PauliTerm::PauliI {
                        continue; // Skip PauliI operations as they don't create interactions
                    }

                    let i_loc = *nodes
                        .get(&Qubit::new(i))
                        .expect("fetching node index in interaction graph");

                    // For each other non-PauliI operation, add an edge between the qubits
                    for (j, term_j) in axis.iter().enumerate() {
                        if i == j || *term_j == PauliTerm::PauliI {
                            continue; // Skip self-connections and PauliI operations
                        }

                        // Get or create the node for qubit j

                        let j_loc = *nodes
                            .get(&Qubit::new(j))
                            .expect("fetching node index in interaction graph");

                        // Update the edges in both directions
                        g.update_edge(i_loc, j_loc, 0);
                        g.update_edge(j_loc, i_loc, 0);
                    }
                }
            }
        }
    }
    return g;
}
pub fn circuit_to_layers(c: &mut Circuit) -> Vec<Vec<Gate>> {
    let mut layers = vec![];
    while !c.gates.is_empty() {
        let l = c.get_front_layer();
        c.remove_gates(&l);
        layers.push(l);
    }
    return layers;
}
pub fn simulated_anneal<T: Clone>(
    start: T,
    initial_temp: f64,
    term_temp: f64,
    cool_rate: f64,
    random_neighbor: impl Fn(&T) -> T,
    cost_function: impl Fn(&T) -> f64,
) -> T {
    let mut best = start.clone();
    let mut best_cost = cost_function(&best);
    let mut current = start.clone();
    let mut curr_cost = cost_function(&current);
    let mut temp = initial_temp;
    while temp > term_temp {
        let next = random_neighbor(&current);
        let next_cost = cost_function(&next);
        let delta_curr = next_cost - curr_cost;
        let delta_best = next_cost - best_cost;
        let rand: f64 = rand::random();
        if delta_best < 0.0 {
            best = next.clone();
            best_cost = next_cost;
            current = next;
            curr_cost = next_cost;
        } else if rand < (-delta_curr / temp).exp() {
            current = next;
            curr_cost = next_cost;
        }
        temp *= cool_rate;
    }
    return best;
}

#[derive(Clone, Copy)]
pub enum Move {
    Swap(Qubit, Qubit),
    IntoOpen(Qubit, Location),
}

fn random_move<A: Architecture>(map: &QubitMap, arch: &A) -> Move {
    let mut moves = vec![];
    for q1 in map.keys() {
        for q2 in map.keys() {
            if q1 == q2 {
                continue;
            }
            moves.push(Move::Swap(*q1, *q2));
        }
    }
    for q in map.keys() {
        for l in arch.locations() {
            if !map.values().any(|x| *x == l) {
                moves.push(Move::IntoOpen(*q, l));
            }
        }
    }
    let rng = &mut rand::rng();
    let chosen_move = *moves.choose(rng).unwrap();
    return chosen_move;
}

pub fn fast_mapping_simulated_anneal<A: Architecture>(
    start: &QubitMap,
    arch: &A,
    initial_temp: f64,
    term_temp: f64,
    cool_rate: f64,
    cost_function: impl Fn(&QubitMap) -> f64,
    delta_on_move: impl Fn(&QubitMap, Move) -> f64,
) -> QubitMap {
    let mut best = start.clone();
    let mut best_cost = cost_function(&best);
    let mut current = start.clone();
    let mut curr_cost = best_cost;
    let mut temp = initial_temp;
    let mut best_to_curr = 0.0;
    while temp > term_temp {
        let next_move = random_move(&current, arch);
        let next: HashMap<Qubit, Location> = match next_move {
            Move::Swap(q1, q2) => {
                let mut new_map = current.clone();
                let loc1 = current.get(&q1).unwrap();
                let loc2 = current.get(&q2).unwrap();
                new_map.insert(q1, *loc2);
                new_map.insert(q2, *loc1);
                new_map
            }
            Move::IntoOpen(qubit, location) => {
                let mut new_map = current.clone();
                new_map.insert(qubit, location);
                new_map
            }
        };
        let delta_curr = delta_on_move(&current, next_move);
        let delta_best = delta_curr + best_to_curr;
        let rand: f64 = rand::random();
        if delta_best < 0.0 {
            best = next.clone();
            best_cost = best_cost + delta_best;
            current = next;
            curr_cost = curr_cost + delta_curr;
            best_to_curr = curr_cost - best_cost;
        } else if rand < (-delta_curr / temp).exp() {
            current = next;
            curr_cost = curr_cost + delta_curr;
            best_to_curr = curr_cost - best_cost;
        }
        temp *= cool_rate;
    }
    return best;
}

pub fn swap_random_array_elements<T: Clone>(array: &Vec<T>) -> Vec<T> {
    let mut rng = rand::rng();

    let idx1 = rng.random_range(0..array.len());
    let mut idx2 = rng.random_range(0..array.len() - 1);

    // Adjust idx2 to ensure it's different from idx1
    if idx2 >= idx1 {
        idx2 += 1;
    }
    let mut new = array.clone();
    new.swap(idx1, idx2);
    return new;
}

pub fn reduced_graph<A: Architecture>(arch: &A) -> Graph<Location, ()> {
    let (full_graph, index_map) = arch.graph();
    let mut reduced_graph = Graph::new();
    let mut reduced_index_map = HashMap::new();
    let locations = arch.locations();
    for l in &locations {
        let n = reduced_graph.add_node(*l);
        reduced_index_map.insert(*l, n);
    }
    for n in full_graph.node_indices() {
        if !locations.contains(&full_graph[n]) {
            let neighbors = full_graph.neighbors(n);

            for (n1, n2) in neighbors.tuple_combinations() {
                if reduced_index_map.contains_key(&full_graph[n1])
                    && reduced_index_map.contains_key(&full_graph[n2])
                {
                    reduced_graph.add_edge(
                        reduced_index_map[&full_graph[n1]],
                        reduced_index_map[&full_graph[n2]],
                        (),
                    );
                    reduced_graph.add_edge(
                        reduced_index_map[&full_graph[n2]],
                        reduced_index_map[&full_graph[n1]],
                        (),
                    );
                }
            }
        }
    }
    return reduced_graph;
}
