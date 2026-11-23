//! Exact play on graphs of randomly generated, unbounded Voronoi maps.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use Colors::*;
use Outcome::{Draw, Loss, Win};



// Exact search cost can grow exponentially with NUM_POINTS.
const NUM_GAMES: usize = 1000;
const NUM_POINTS: usize = 10;

// None chooses a random seed once per run. Some(42) replays a batch of maps.
// Each game uses its index together with this seed to generate its sites.
const MAP_SEED: Option<u64> = None;

// Add a variant here to make a new color available to any palette.
#[allow(dead_code, reason = "each experiment can use a subset of the colors")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Colors {
	Red,
	Yellow,
	Green,
	Blue,
	// Purple,
}

// One palette per player, in turn order. Player 1 opens every game.
const AVAILABLE_COLORS: &[&[Colors]] = &[
	&[Red, Yellow, Green, Blue],
	&[Red, Yellow],
	// &[Red, Yellow, Green, Blue],

	// &[Red, Yellow],
	// &[Green, Blue],

	// &[Red, Green, Blue],
	// &[Red, Green, Yellow],

	// &[Yellow, Red],
	// &[Yellow, Green],
	// &[Yellow, Blue],
];

const NUM_PLAYERS: usize = AVAILABLE_COLORS.len();

// Each row gives every player's outcome, in seat order.
// ON_FULL_MAP uses the row of the player who colors the last cell.
// ON_NO_MOVE uses the row of the player unable to move on an incomplete map.
// by_actor assigns its first outcome to that player and its second to the others.
type OutcomeTable = [[Outcome; NUM_PLAYERS]; NUM_PLAYERS];

// To give player 2 every completion in a two-player game, use [[Loss, Win]; 2].
// With three players, award a win only to the one who blocked the next player
// by setting ON_NO_MOVE to [[Loss, Loss, Win], [Win, Loss, Loss], [Loss, Win, Loss]].

// The last legal move loses, whether it fills the map or blocks the opponent.
// With 10 sites, all four colors for P1 and red/yellow for P2, P1 won 50,350
// of 100,000 maps (50,000 each with seeds 20_261_123 and 8_675_309).
const ON_FULL_MAP: OutcomeTable = by_actor(Loss, Win);
const ON_NO_MOVE: OutcomeTable = by_actor(Win, Loss);
// const ON_FULL_MAP: OutcomeTable = by_actor(Win, Loss); // last cell wins
// const ON_FULL_MAP: OutcomeTable = by_actor(Draw, Draw); // draw on completion
// const ON_NO_MOVE: OutcomeTable = by_actor(Loss, Win); // blocked player loses
// const ON_NO_MOVE: OutcomeTable = [[Loss, Loss, Win], [Win, Loss, Loss], [Loss, Win, Loss]]; // for 3 players



const _: () = {
	assert!(NUM_GAMES > 0, "at least one game is needed for percentages");
	assert!(NUM_POINTS > 0, "a map needs at least one site");
	assert!(NUM_PLAYERS > 0, "at least one player must take turns");
};



fn main() {
	let seed = MAP_SEED.unwrap_or_else(rand::random);
	let progress = game_progress(NUM_GAMES);
	let totals = analyze::<NUM_POINTS, NUM_PLAYERS>(NUM_GAMES, seed, &RULES, progress);
	println!("Solved {NUM_GAMES} games with {NUM_POINTS} cells each (map seed {seed}, part of total seed).");
	let percent = |count| 100.0 * (count as f64) / (NUM_GAMES as f64);
	for (i, tally) in totals.into_iter().enumerate() {
		let [wins, losses, draws] = [Win, Loss, Draw].map(|outcome| tally[outcome as usize]);
		println!(
			"Player {}   {wins} wins ({:.2}%)   {losses} losses ({:.2}%)   {draws} draws ({:.2}%)",
			i + 1,
			percent(wins),
			percent(losses),
			percent(draws)
		);
	}
}



fn analyze<const N: usize, const P: usize>(
	games: usize,
	seed: u64,
	rules: &Rules<P>,
	on_game_finished: impl Fn() + Sync,
) -> [Tally; P] {
	(0..games)
		.into_par_iter()
		.map(|game| {
			let sites = gaussian_sites::<N>(&mut map_rng(seed, game));
			let diagram = VoronoiDiagram::from_sites(sites);
			let graph = diagram.cell_graph();
			solve(&graph, rules)
		})
		.fold(
			|| [Tally::default(); P],
			|mut totals, game| {
				for (tally, outcome) in totals.iter_mut().zip(game) {
					tally[outcome as usize] += 1;
				}
				on_game_finished();
				totals
			},
		)
		.reduce(
			|| [Tally::default(); P],
			|mut left, right| {
				for (tally, other) in left.iter_mut().zip(right) {
					for (count, add) in tally.iter_mut().zip(other) {
						*count += add;
					}
				}
				left
			},
		)
}

fn game_progress(total: usize) -> impl Fn() + Sync {
	let started = Instant::now();
	let state = io::stderr().is_terminal().then(|| {
		let _ = draw_game_progress(io::stderr().lock(), 0, total, Duration::ZERO);
		Mutex::new((0_usize, started))
	});

	move || {
		let Some(state) = &state else { return };
		let mut state = state.lock().unwrap();
		let (completed, last_draw) = &mut *state;
		*completed += 1;
		let now = Instant::now();
		if *completed == total || now.duration_since(*last_draw) >= Duration::from_millis(100) {
			*last_draw = now;
			let _ = draw_game_progress(io::stderr().lock(), *completed, total, now - started);
		}
	}
}

fn draw_game_progress(
	mut out: impl Write,
	completed: usize,
	total: usize,
	elapsed: Duration,
) -> io::Result<()> {
	const WIDTH: usize = 20;
	let percent = completed as u128 * 100 / total as u128;
	let filled = percent as usize * WIDTH / 100;
	let clock = |seconds: u64| format!("{:02}:{:02}", seconds / 60, seconds % 60);
	let (rate, eta) = if completed == 0 || elapsed.is_zero() {
		(0.0, String::from("--:--"))
	} else {
		let rate = completed as f64 / elapsed.as_secs_f64();
		(rate, clock(((total - completed) as f64 / rate).ceil() as u64))
	};
	write!(
		out,
		"\rGames [{}{}] {percent:3}% {completed}/{total}  {rate:.1}/s  elapsed {}  ETA {eta}\x1b[K",
		"#".repeat(filled),
		"-".repeat(WIDTH - filled),
		clock(elapsed.as_secs()),
	)?;
	if completed == total {
		writeln!(out)?;
	}
	out.flush()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
	Loss,
	Draw,
	Win,
}

struct Rules<'a, const P: usize> {
	palettes: &'a [&'a [Colors]],
	on_full_map: [[Outcome; P]; P],
	on_no_move: [[Outcome; P]; P],
}

const RULES: Rules<'static, NUM_PLAYERS> = Rules {
	palettes: AVAILABLE_COLORS,
	on_full_map: ON_FULL_MAP,
	on_no_move: ON_NO_MOVE,
};

fn map_rng(seed: u64, game: usize) -> StdRng {
	// Keep the experiment and map index in separate parts of the key so that
	// incrementing the experiment seed gives a fresh set of maps.
	let mut key = [0; 32];
	key[..8].copy_from_slice(&seed.to_le_bytes());
	key[8..16].copy_from_slice(&(game as u64).to_le_bytes());
	StdRng::from_seed(key)
}



type Point = [f64; 2];
type Graph<const N: usize> = [Vec<usize>; N];

fn gaussian_sites<const N: usize>(rng: &mut impl Rng) -> [Point; N] {
	let mut sites = [[0.0; 2]; N];
	for i in 0..N {
		loop {
			let point = [StandardNormal.sample(rng), StandardNormal.sample(rng)];
			if !sites[..i].contains(&point) {
				sites[i] = point;
				break;
			}
		}
	}
	sites
}



const fn by_actor<const P: usize>(actor: Outcome, others: Outcome) -> [[Outcome; P]; P] {
	let mut rows = [[others; P]; P];
	let mut i = 0;
	while i < P {
		rows[i][i] = actor;
		i += 1;
	}
	rows
}



// On the bisector of sites a and b, a point is midpoint(a,b) + t * perp(b-a).
// This interval represents a segment, ray, whole line, or a single vertex.
#[derive(Debug)]
struct VoronoiContact {
	cells: [usize; 2],
	start: f64,
	end: f64,
}

struct VoronoiDiagram<const N: usize> {
	sites: [Point; N],
	contacts: Vec<VoronoiContact>,
}

impl<const N: usize> VoronoiDiagram<N> {
	fn from_sites(sites: [Point; N]) -> Self {
		let mut contacts = Vec::new();
		for a in 0..N {
			for b in a + 1..N {
				if let Some(contact) = Self::contact(&sites, a, b) {
					contacts.push(contact);
				}
			}
		}
		Self { sites, contacts }
	}

	fn contact(sites: &[Point; N], a: usize, b: usize) -> Option<VoronoiContact> {
		let [ax, ay] = sites[a];
		let [bx, by] = sites[b];
		let (dx, dy) = (bx - ax, by - ay);
		let (mut start, mut end) = (f64::NEG_INFINITY, f64::INFINITY);

		// Intersect the bisector with each other site's nearest-site half-plane.
		// For site k, distance(p,a)^2 <= distance(p,k)^2 reduces to
		// 2 * cross(b-a,k-a) * t <= dot(k-a,k-b).
		for (k, &[x, y]) in sites.iter().enumerate() {
			if k == a || k == b {
				continue;
			}
			let (kx, ky) = (x - ax, y - ay);
			let slope = 2.0 * (dx * ky - dy * kx);
			let bound = kx * (x - bx) + ky * (y - by);
			if slope > 0.0 {
				end = end.min(bound / slope);
			} else if slope < 0.0 {
				start = start.max(bound / slope);
			} else if bound < 0.0 {
				// This collinear site is closer everywhere on the bisector.
				return None;
			}
			if start > end {
				return None;
			}
		}
		Some(VoronoiContact {
			cells: [a, b],
			start,
			end,
		})
	}

	fn cell_graph(&self) -> Graph<N> {
		let mut graph = self.sites.each_ref().map(|_| Vec::new());
		for contact in &self.contacts {
			// A vertex shared by cocircular cells has zero border length.
			if contact.start < contact.end {
				let [a, b] = contact.cells;
				graph[a].push(b);
				graph[b].push(a);
			}
		}
		graph
	}
}



fn solve<const N: usize, const P: usize>(graph: &Graph<N>, rules: &Rules<P>) -> [Outcome; P] {
	best_outcome(graph, rules, &mut [None; N], 0, &mut HashMap::new())
}

// Each mover ranks Win > Draw > Loss for their own seat (max-n). This is
// minimax for two players with opposing outcomes. Equal moves keep the first
// result in site-placement order, then in the mover's listed palette order.
fn best_outcome<const N: usize, const P: usize>(
	graph: &Graph<N>,
	rules: &Rules<P>,
	board: &mut [Option<Colors>; N],
	played: usize,
	memo: &mut HashMap<[Option<Colors>; N], [Outcome; P]>,
) -> [Outcome; P] {
	// Every move fills exactly one cell, so the board also determines the turn.
	if let Some(&result) = memo.get(board) {
		return result
	}
	let player = played % P;
	let mut best: Option<[Outcome; P]> = None;
	'moves: for (cell, neighbors) in graph.iter().enumerate() {
		if board[cell].is_some() {
			continue
		}
		for &color in rules.palettes[player] {
			if neighbors.iter().any(|&other| board[other] == Some(color)) {
				continue
			}
			if played + 1 == N {
				return rules.on_full_map[player]
			}
			board[cell] = Some(color);
			let result = best_outcome(graph, rules, board, played + 1, memo);
			board[cell] = None;
			if best.is_none_or(|previous| result[player] > previous[player]) {
				best = Some(result);
				if result[player] == Win {
					break 'moves
				}
			}
		}
	}
	let result = best.unwrap_or(rules.on_no_move[player]);
	memo.insert(*board, result);
	result
}

// Each seat's counts are indexed by Outcome.
type Tally = [usize; 3];





#[cfg(test)]
mod tests {
	use super::*;

	fn usual_rules<'a, const P: usize>(palettes: &'a [&'a [Colors]; P]) -> Rules<'a, P> {
		Rules {
			palettes,
			on_full_map: by_actor(Win, Loss),
			on_no_move: by_actor(Loss, Win),
		}
	}

	#[test]
	fn sites_use_two_independent_standard_normal_samples_in_placement_order() {
		let mut rng = StdRng::seed_from_u64(73);
		let sites = gaussian_sites::<12>(&mut rng);
		let mut replay = StdRng::seed_from_u64(73);
		for (i, site) in sites.iter().enumerate() {
			let expected: Point = [
				StandardNormal.sample(&mut replay),
				StandardNormal.sample(&mut replay),
			];
			assert_eq!(*site, expected);
			assert!(!sites[..i].contains(site));
		}
	}

	#[test]
	fn map_streams_repeat_by_seed_and_index_without_shifting_between_experiments() {
		let sites = |seed, index| gaussian_sites::<4>(&mut map_rng(seed, index));
		assert_eq!(sites(42, 1), sites(42, 1));
		assert_ne!(sites(42, 0), sites(42, 1));
		assert_ne!(sites(42, 1), sites(43, 0));
	}

	#[test]
	fn collinear_sites_have_whole_line_borders_between_consecutive_cells() {
		let diagram = VoronoiDiagram::from_sites([[2.0, 3.0], [0.0, 1.0], [1.0, 2.0]]);
		assert_eq!(diagram.cell_graph(), [vec![2], vec![2], vec![0, 1]]);
		assert!(
			diagram
				.contacts
				.iter()
				.all(|c| c.start == f64::NEG_INFINITY && c.end == f64::INFINITY)
		);
		assert_eq!(
			VoronoiDiagram::from_sites([[0.0, 0.0]]).cell_graph(),
			[Vec::<usize>::new()]
		);
	}

	#[test]
	fn cocircular_cells_sharing_only_a_vertex_are_not_neighbors() {
		let diagram = VoronoiDiagram::from_sites([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
		assert_eq!(
			diagram.cell_graph(),
			[vec![1, 3], vec![0, 2], vec![1, 3], vec![0, 2]]
		);
		let diagonal = diagram.contacts.iter().find(|c| c.cells == [0, 2]).unwrap();
		assert_eq!(diagonal.start, diagonal.end);
	}

	#[test]
	fn a_hull_border_can_start_far_outside_the_sites() {
		let diagram = VoronoiDiagram::from_sites([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.000001]]);
		let far_ray = diagram.contacts.iter().find(|c| c.cells == [0, 1]).unwrap();
		assert_eq!(far_ray.start, f64::NEG_INFINITY);
		assert!(far_ray.end < -100_000.0);
		assert_eq!(diagram.cell_graph(), [vec![1, 2], vec![0, 2], vec![0, 1]]);
	}

	#[test]
	fn best_play_blocks_the_opponent_by_taking_the_middle_cell() {
		let path = [vec![1], vec![0, 2], vec![1]];
		let rules = usual_rules(&[&[Red], &[Red]]);
		let mut opening_at_end = [Some(Red), None, None];
		assert_eq!(
			best_outcome(&path, &rules, &mut opening_at_end, 1, &mut HashMap::new()),
			[Loss, Win]
		);
		assert_eq!(solve(&path, &rules), [Win, Loss]);
	}

	#[test]
	fn the_player_filling_the_last_cell_selects_the_completion_row() {
		let mut rules = usual_rules(&[&[Red], &[Blue]]);
		rules.on_full_map = [[Loss, Win], [Draw, Draw]];
		assert_eq!(solve(&[vec![]], &rules), [Loss, Win]);
		assert_eq!(solve(&[vec![1], vec![0]], &rules), [Draw, Draw]);
	}

	#[test]
	fn a_fifth_color_can_make_an_otherwise_blocked_move_legal() {
		let diagram = VoronoiDiagram::from_sites([
			[-1.0, -1.0],
			[1.0, -1.0],
			[1.0, 1.0],
			[-1.0, 1.0],
			[0.0, 0.0],
		]);
		let graph = diagram.cell_graph();
		assert_eq!(graph[4], vec![0, 1, 2, 3]);
		assert!(
			diagram
				.contacts
				.iter()
				.filter(|c| c.cells[1] == 4)
				.all(|c| c.start.is_finite() && c.end.is_finite())
		);
		let mut board = [Some(Red), Some(Yellow), Some(Green), Some(Blue), None];
		let four = usual_rules(&[&[Red, Yellow, Green, Blue], &[Red, Yellow, Green, Blue]]);
		assert_eq!(
			best_outcome(&graph, &four, &mut board, 4, &mut HashMap::new()),
			[Loss, Win]
		);
		// let five = usual_rules(&[
		// 	&[Red, Yellow, Green, Blue, Purple],
		// 	&[Red, Yellow, Green, Blue],
		// ]);
		// assert_eq!(
		// 	best_outcome(&graph, &five, &mut board, 4, &mut HashMap::new()),
		// 	[Win, Loss]
		// );
	}

	#[test]
	fn multiplayer_search_prefers_a_win_then_a_draw_and_breaks_ties_in_palette_order() {
		let edge = [vec![1], vec![0]];
		let mut rules = usual_rules(&[&[Blue, Red], &[Red], &[Green]]);
		// Opening with blue lets player 2 fill the other cell. Red blocks them.
		rules.on_full_map[1] = [Win, Win, Loss];
		assert_eq!(solve(&edge, &rules), [Win, Win, Loss]);
		rules.palettes = &[&[Red, Blue], &[Red], &[Green]];
		assert_eq!(solve(&edge, &rules), [Win, Loss, Win]);
		rules.on_no_move[1] = [Draw, Loss, Win];
		assert_eq!(solve(&edge, &rules), [Win, Win, Loss]);
		rules.on_full_map[1] = [Loss, Win, Loss];
		assert_eq!(solve(&edge, &rules), [Draw, Loss, Win]);
	}

	#[test]
	fn player_two_in_a_three_player_game_can_prefer_a_shared_win_with_player_one() {
		let path = [vec![1], vec![0, 2], vec![1, 3], vec![2]];
		let mut rules = usual_rules(&[&[Blue], &[Red], &[Red]]);
		rules.on_full_map[0] = [Loss, Draw, Win];
		let mut board = [Some(Blue), None, None, None];
		// Red in cell 1 or 3 allows completion. Red in cell 2 blocks player 3.
		assert_eq!(
			best_outcome(&path, &rules, &mut board, 1, &mut HashMap::new()),
			[Win, Win, Loss]
		);
	}

	// Independent two-player oracle, from player 1's perspective. Expand every
	// move and take the minimum on player 2's turns.
	fn exhaustive_minimax<const N: usize>(
		graph: &Graph<N>,
		rules: &Rules<2>,
		board: [Option<Colors>; N],
	) -> Outcome {
		let played = board.iter().flatten().count();
		if played == N {
			return rules.on_full_map[(played - 1) % 2][0];
		}
		let player = played % 2;
		let mut children = Vec::new();
		for cell in 0..N {
			for &color in rules.palettes[player] {
				if board[cell].is_none() && graph[cell].iter().all(|&n| board[n] != Some(color)) {
					let mut next = board;
					next[cell] = Some(color);
					children.push(exhaustive_minimax(graph, rules, next));
				}
			}
		}
		if player == 0 {
			children.into_iter().max()
		} else {
			children.into_iter().min()
		}
		.unwrap_or(rules.on_no_move[player][0])
	}

	#[test]
	fn cached_solver_matches_exhaustive_minimax_on_all_four_cell_graphs() {
		let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
		let shared = usual_rules(&[&[Red, Yellow, Green, Blue], &[Red, Yellow, Green, Blue]]);
		let mut overlapping = usual_rules(&[&[Red, Yellow], &[Yellow, Blue]]);
		overlapping.on_full_map = by_actor(Draw, Draw);
		for mask in 0..(1 << pairs.len()) {
			let mut graph: Graph<4> = std::array::from_fn(|_| Vec::new());
			for (bit, &(a, b)) in pairs.iter().enumerate() {
				if mask & (1 << bit) != 0 {
					graph[a].push(b);
					graph[b].push(a);
				}
			}
			for rules in [&shared, &overlapping] {
				assert_eq!(
					solve(&graph, rules)[0],
					exhaustive_minimax(&graph, rules, [None; 4]),
					"graph mask {mask}"
				);
			}
		}
	}

	#[test]
	fn statistics_count_every_game_for_every_seat_including_an_immediate_block() {
		use std::sync::atomic::{AtomicUsize, Ordering};

		let mut rules = usual_rules(&[&[], &[Red], &[Blue]]);
		rules.on_no_move[0] = [Loss, Draw, Win];
		let finished = AtomicUsize::new(0);
		let totals = analyze::<3, 3>(17, 7, &rules, || {
			finished.fetch_add(1, Ordering::Relaxed);
		});
		assert_eq!(finished.into_inner(), 17);
		for (tally, outcome) in totals.into_iter().zip([Loss, Draw, Win]) {
			assert_eq!(tally[outcome as usize], 17);
			assert_eq!(tally.iter().sum::<usize>(), 17);
		}
	}

	#[test]
	fn progress_redraws_one_line_and_finishes_at_one_hundred_percent() {
		let mut output = Vec::new();
		draw_game_progress(&mut output, 0, 10, Duration::ZERO).unwrap();
		draw_game_progress(&mut output, 4, 10, Duration::from_secs(8)).unwrap();
		draw_game_progress(&mut output, 10, 10, Duration::from_secs(20)).unwrap();
		let output = String::from_utf8(output).unwrap();
		let frames: Vec<_> = output.split_terminator('\r').skip(1).collect();
		assert_eq!(frames.len(), 3);
		assert!(frames[0].contains("0% 0/10") && frames[0].contains("ETA --:--"));
		assert!(frames[1].contains("40% 4/10  0.5/s  elapsed 00:08  ETA 00:12"));
		assert!(frames[2].contains("100% 10/10") && frames[2].contains("ETA 00:00"));
		assert_eq!(output.lines().count(), 1);
		assert!(output.ends_with('\n'));
	}
}

