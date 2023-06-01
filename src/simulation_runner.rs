use polars::prelude::*;

use crate::bernulli_bandit::generate_uniform_random_number;
use crate::bernulli_multi_armed_bandits_game::BernulliMultiArmedBanditsGame;

use crate::constants::{
    NUM_OF_GAMES_TO_PLAY,
    IS_VERBOSE_MODE,
    EPSILON,
    ALPHA,
    NUM_OF_BANDITS,
    NUM_OF_TURNS_IN_A_GAME,
};

pub struct SimulationRunner {
    num_of_games: i32,
    games: Vec<BernulliMultiArmedBanditsGame>,
    df_vector: Option<Vec<DataFrame>>,
    resulting_actions_vector: Option<Vec<Vec<usize>>>,
    resulting_rewards_vector: Option<Vec<Vec<f64>>>,
    total_rewards_per_game: Option<Vec<f64>>,
}

impl SimulationRunner {
    /// Initalizes a new simulation
    pub fn new() -> Self {
        let mut games = vec![];
        for index in 0..NUM_OF_GAMES_TO_PLAY as usize {
            games.insert(index, BernulliMultiArmedBanditsGame::new());
        }
        SimulationRunner {
            num_of_games: NUM_OF_GAMES_TO_PLAY,
            games,
            df_vector: None,
            resulting_actions_vector: None,
            resulting_rewards_vector: None,
            total_rewards_per_game: None,
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

    /// Runs Bernulli Multi Armed Bandit simulation
    pub fn run_bernulli_multi_armed_simulation_game(&mut self) {
        // Initalize data recorders
        let mut dataframes: Vec<DataFrame> = Vec::new();
        let mut resulting_actions: Vec<Vec<usize>> = Vec::new();
        let mut resulting_rewards: Vec<Vec<f64>> = Vec::new();
        let mut average_rewards: Vec<f64> = Vec::new();
        let mut total_rewards = Vec::new();

        // Run each game and record results for each game run.
        for game_run in 0..self.num_of_games as usize {
            if IS_VERBOSE_MODE {
                println!("\n# Game run: {} #", game_run);
            }

            let game = &mut self.games[game_run];

            game.run_game();

            // Record results
            average_rewards.push(game.calculate_mean_reward());
            dataframes.push(game.df_results.clone().unwrap());
            resulting_actions.push(game.resulting_actions.clone().unwrap());
            resulting_rewards.push(game.resulting_rewards.clone().unwrap());
            total_rewards.push(game.calculate_total_reward());

            // Summarise each game run.
            if IS_VERBOSE_MODE {
                println!(
                    "Epsilon: {} \t Alpha: {} \t Average Mean Reward: {:?} \t total_reward: {}",
                    self.games[game_run].get_epsilon(),
                    self.games[game_run].get_alpha(),
                    self.games[game_run].calculate_mean_reward(),
                    self.games[game_run].calculate_total_reward()
                );
            }
        }

        // Record results for all the games:
        // 1) resulting dataframe
        // 2) resulting actions taken
        // 3) resulting rewards recieved
        // 4) total rewards recieved for each game
        self.df_vector = Some(dataframes);
        self.resulting_actions_vector = Some(resulting_actions);
        self.resulting_rewards_vector = Some(resulting_rewards);
        self.total_rewards_per_game = Some(total_rewards);

        if IS_VERBOSE_MODE {
            self.print_df_results();
            println!("Average rewards per game is: {:?}", average_rewards);
        }

        // Calculate mean reward for all the games
        let total_rewards: f64 = average_rewards.iter().sum();
        let mean_reward: f64 = total_rewards / (self.num_of_games as f64);

        // We expect the average mean reward to be close to 0.5 in case all random actions
        // have been selected in all the steps. We also have a random probabilities for each
        // of the armed bandit.
        println!("## Mean for all the games played ##");
        println!("\nMean Reward: {:?} \t total_reward: {}", mean_reward, total_rewards);
    }

    pub fn bulk_run_bernulli_simulation(&mut self) {
        println!("### Bulk running Bernulli Multi Armed Bandits Simulation ###");
        println!(
            "Number of bandits: {} \t Number of turns in the game: {} \t Number of games to play: {}",
            NUM_OF_BANDITS,
            NUM_OF_TURNS_IN_A_GAME,
            NUM_OF_GAMES_TO_PLAY
        );
        // Generate vector with random epsilons and alphas different for each game
        let epsilons = vec![0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let alphas = vec![0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        // let epsilons: Vec<f64> = (0..self.num_of_games)
        //     .map(|_| generate_uniform_random_number())
        //     .collect();
        // let alphas: Vec<f64> = (0..self.num_of_games)
        //     .map(|_| generate_uniform_random_number())
        //     .collect();

        self.games = alphas
            .iter()
            .zip(epsilons.iter())
            .map(|(&alpha, &epsilon)| {
                let mut game = BernulliMultiArmedBanditsGame::new();
                game.set_alpha(alpha);
                game.set_epsilon(epsilon);
                game
            })
            .collect();

        // Run games and save results
        self.run_bernulli_multi_armed_simulation_game();
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
