use polars::prelude::*;

use crate::{ bernulli_multi_armed_bandits_game::BernulliMultiArmedBanditsGame };

use crate::constants::NUM_OF_GAMES_TO_RUN;

pub struct SimulationRunner {
    pub game: BernulliMultiArmedBanditsGame,
    num_of_games: i32,
    statistics: Option<Vec<DataFrame>>,
}

impl SimulationRunner {
    pub fn new_bernulli_bandit() -> Self {
        SimulationRunner {
            game: BernulliMultiArmedBanditsGame::new(),
            num_of_games: NUM_OF_GAMES_TO_RUN,
            statistics: None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_creation_of_game() {
        let mut runner = SimulationRunner::new_bernulli_bandit();

        runner.game.run_game();
    }
}
