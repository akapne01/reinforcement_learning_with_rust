mod bernulli_bandit;
mod bandits_game;

fn main() {
    let no_of_bandits: usize = 10;
    let no_of_trials = 100_000;
    bandits_game::run_with(no_of_bandits, no_of_trials);
}
