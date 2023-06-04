use std::sync::{ Arc, Mutex };
use std::thread;

use bernoulli_multi_armed_bandits_game::BernoulliGameLearningRunner;

mod constants;
#[path = "environments/bernoulli_bandit.rs"]
mod bernoulli_bandit;
mod bernoulli_multi_armed_bandits_game;
mod statistics_calculator;

fn main() {
    let mut runner = BernoulliGameLearningRunner::new();
    runner.run_all_games();

    let mut runner_2 = BernoulliGameLearningRunner::new();
    runner_2.run_all_games_in_parallel();
}
