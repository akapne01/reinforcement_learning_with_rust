mod bernulli_bandit;
mod bandits_game;

use crate::bandits_game::BanditsGame;

fn main() {
    let no_of_bandits: usize = 10;
    let no_of_trials = 10000;
    let mut game = BanditsGame::new(no_of_bandits as i32, no_of_trials);
    game.run_stochastic();
    game.print_statistics();
}
