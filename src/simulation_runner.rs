use polars::prelude::*;

use crate::{ bernulli_multi_armed_bandits_game::BernulliMultiArmedBanditsGame };

use crate::constants::{ NUM_OF_GAMES_TO_PLAY, IS_VERBOSE_MODE, EPSILON, ALPHA };

pub struct SimulationRunner {
    num_of_games: i32,
    df_vector: Option<Vec<DataFrame>>,
    resulting_actions_vector: Option<Vec<Vec<usize>>>,
    resulting_rewards_vector: Option<Vec<Vec<f64>>>,
}

impl SimulationRunner {
    pub fn new() -> Self {
        SimulationRunner {
            num_of_games: NUM_OF_GAMES_TO_PLAY,
            df_vector: None,
            resulting_actions_vector: None,
            resulting_rewards_vector: None,
        }
    }

    fn print_df_results(&mut self) {
        match &self.df_vector {
            Some(list) => {
                println!("### Results of Bernulli Multi Armed Simulation ###");

                for (game_number, df) in list.into_iter().enumerate() {
                    println!("### Results Dataframe for trial : {} ###", game_number);
                    println!("{:?} \n", df);
                    println!("Epsilon: {} \t Alpha: {}", EPSILON, ALPHA);
                }
            }
            None => {
                self.run_bernulli_multi_armed_simulation_game();
                self.print_df_results();
            }
        }
    }

    pub fn run_bernulli_multi_armed_simulation_game(&mut self) {
        let mut dataframes: Vec<DataFrame> = vec![];
        let mut resulting_actions_vector: Vec<Vec<usize>> = vec![];
        let mut resulting_rewards_vector: Vec<Vec<f64>> = vec![];
        let mut average_rewards: Vec<f64> = vec![];
        for game_run in 0..self.num_of_games as usize {
            if IS_VERBOSE_MODE {
                println!("\n# Game run: {} #", game_run);
            }
            let mut game = BernulliMultiArmedBanditsGame::new();
            game.run_game();
            average_rewards.insert(game_run, game.calculate_average_reward());
            dataframes.insert(game_run, game.df_results.unwrap());
            resulting_actions_vector.insert(game_run, game.resulting_actions.unwrap());
            resulting_rewards_vector.insert(game_run, game.resulting_rewards.unwrap());
        }

        self.df_vector = Some(dataframes);
        self.resulting_actions_vector = Some(resulting_actions_vector);
        self.resulting_rewards_vector = Some(resulting_rewards_vector);
        self.print_df_results();
        println!("Average rewards per game is: {:?}", average_rewards);
        let total_rewards: f64 = average_rewards.iter().sum();
        let mean_reward: f64 = total_rewards / (self.num_of_games as f64);
        // We expect the average mean reward to be close to 0.5 in case all random actions
        // have been selected in all the steps. We also have a random probabilities for each
        // of the armed bandit.
        println!(
            "Epsilon: {} \t Alpha: {} \t Average Mean Reward: {:?}",
            EPSILON,
            ALPHA,
            mean_reward
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_creation_of_game() {
        let mut runner = SimulationRunner::new();
        assert!(runner.df_vector.is_none());

        runner.run_bernulli_multi_armed_simulation_game();

        assert!(runner.df_vector.is_some());
    }
}
