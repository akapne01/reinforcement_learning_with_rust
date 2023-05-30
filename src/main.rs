mod bernulli_bandit;
mod bernulli_multi_armed_bandits_game;

fn main() {
    let no_of_bandits: usize = 10;
    let no_of_trials = 100_000;
    bernulli_multi_armed_bandits_game::run_with(no_of_bandits, no_of_trials);
}
