mod constants;
mod bernulli_bandit;
mod bernulli_multi_armed_bandits_game;
mod simulation_runner;
use crate::simulation_runner::SimulationRunner;

fn main() {
    let mut runner = SimulationRunner::new();
    runner.run_bernulli_multi_armed_simulation_game();
}
