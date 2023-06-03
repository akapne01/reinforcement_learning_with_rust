use statistics_calculator::BernoulliAgentStatisticsWrapper;

mod constants;
#[path = "environments/bernoulli_bandit.rs"]
mod bernoulli_bandit;
mod bernoulli_multi_armed_bandits_game;
mod statistics_calculator;

fn main() {
    let mut game = BernoulliAgentStatisticsWrapper::new();
    game.run();
}
