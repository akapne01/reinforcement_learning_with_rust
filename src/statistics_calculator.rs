use polars::prelude::*;

use crate::{
    bernoulli_multi_armed_bandits_game::BernoulliParallelGameRunner,
    constants::{ NUM_OF_GAMES_TO_PLAY, NUM_OF_TURNS_IN_A_GAME, EPSILON, ALPHA },
};

pub struct BernoulliAgentStatisticsWrapper {
    game_runner: BernoulliParallelGameRunner,
    for_game_df: Vec<Option<DataFrame>>,
    df: Option<DataFrame>,
}

impl BernoulliAgentStatisticsWrapper {
    pub fn new() -> Self {
        let game_runner = BernoulliParallelGameRunner::new();
        let n: usize = game_runner.num_of_games;
        BernoulliAgentStatisticsWrapper {
            game_runner,
            for_game_df: vec![None; n],
            df: None,
        }
    }

    pub fn from(runner: BernoulliParallelGameRunner) -> Self {
        let n = runner.num_of_games;
        BernoulliAgentStatisticsWrapper {
            game_runner: runner,
            for_game_df: vec![None; n],
            df: None,
        }
    }

    pub fn run(&mut self) {
        self.game_runner.run_all_games_in_parallel();
        self.save_df_per_each_game();
        self.save_df_for_all_games();
        self.display_statistics();
    }

    /// Populates dataframe for each game separately
    fn save_df_per_each_game(&mut self) {
        std::env::set_var("POLARS_FMT_MAX_COLS", "12");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "12");

        for n in 0..self.game_runner.num_of_games {
            self.populate_dataframe_for_game(n);
        }
    }

    fn save_df_for_all_games(&mut self) {
        // rows: game
        // columns: mean_reward, total_reward, correctness_score
        let mut means = Vec::new();
        let mut scores = Vec::new();
        let mut totals = Vec::new();

        (0..self.game_runner.num_of_games).for_each(|n| {
            let mean_reward = self.get_mean_reward_per_game(n);
            let learning_score = self.get_learning_score_for_game(n);
            let total_reward = self.get_total_reward_per_game(n);

            means.push(mean_reward);
            scores.push(learning_score);
            totals.push(total_reward);
        });

        let df = DataFrame::new(
            vec![
                Series::new("game_number", Vec::from_iter(0..self.game_runner.num_of_games as u32)),
                Series::new("mean_reward", &means),
                Series::new("learning_score", &scores),
                Series::new("total_reward", &totals)
            ]
        ).expect("Failed to create DataFrame");
        self.df = Some(df);
    }

    /// This method writes results in the dataframe and adds additional calculations.
    /// If results are not present, then will run to create it.
    fn populate_dataframe_for_game(&mut self, n: usize) {
        let game = self.game_runner.games[n].clone();
        let num_of_bandits = game.num_of_bandits;

        if game.resulting_actions.is_none() || game.resulting_rewards.is_none() {
            self.game_runner.run_all_games_in_parallel();
        }

        let mut bandits_frequency = vec![0; num_of_bandits];
        let mut bandits_rewards = vec![0.0; num_of_bandits];

        for (&action, &reward) in game.resulting_actions
            .as_ref()
            .unwrap()
            .iter()
            .zip(game.resulting_rewards.as_ref().unwrap().iter()) {
            bandits_frequency[action] += 1;
            bandits_rewards[action] += reward;
        }

        let mut df = DataFrame::new(
            vec![
                Series::new("bandit", Vec::from_iter(0..num_of_bandits as u32)),
                Series::new("actual_probability", &self._get_actual_probabilities_per_game(n)),
                Series::new("learned_probability", &self.game_runner.games[n].q_values),
                Series::new("frequency", &bandits_frequency),
                Series::new("total_reward", &bandits_rewards)
            ]
        ).expect("Failed to create DataFrame");

        df = df
            .lazy()
            .with_column((col("total_reward") / col("frequency")).alias("mean_reward"))
            .collect()
            .unwrap();

        df = df
            .lazy()
            .with_column((col("actual_probability") - col("mean_reward")).alias("diff_actual_mean"))
            .collect()
            .unwrap();

        df = df
            .lazy()
            .with_column(
                (col("actual_probability") - col("learned_probability")).alias(
                    "diff_actual_learned"
                )
            )
            .collect()
            .unwrap();

        df = df
            .lazy()
            .with_column(
                (col("mean_reward") - col("learned_probability")).alias("diff_mean_learned")
            )
            .collect()
            .unwrap();

        df = df.sort(["actual_probability"], true).expect("Couldn't sort the dataframe");
        self.for_game_df[n] = Some(df);
    }

    /// Calculates the average reward received in each turn
    fn get_mean_reward_per_game(&mut self, n: usize) -> f64 {
        let game = self.game_runner.games[n].clone();
        if game.resulting_rewards.is_none() {
            self.game_runner.run_all_games_in_parallel();
        }
        let total: f64 = game.resulting_rewards.as_ref().unwrap().iter().sum();
        total / (game.num_of_turns as f64)
    }

    /// Helper method that allows to obtain the actula probabilities that each arm bandit had
    /// for statistical purposes only.
    /// The mean rewards coverge over time and with many trials to the actual proababilities reflecting
    /// that playing agent has learned them.
    fn _get_actual_probabilities_per_game(&self, n: usize) -> Vec<f64> {
        let game = self.game_runner.games[n].clone();
        let mut probabilities = vec![0.0; game.num_of_bandits as usize];
        for (index, bandit) in game.environment.iter().enumerate() {
            probabilities[index] = bandit._get_actual_probablity();
        }
        probabilities
    }

    /// Calculates the total rewards received by playing game
    fn get_total_reward_per_game(&mut self, n: usize) -> f64 {
        let game = self.game_runner.games[n].clone();
        if game.resulting_rewards.is_none() {
            self.game_runner.run_all_games_in_parallel();
        }
        let mut total = 0.0;
        for reward in game.resulting_rewards.as_ref().expect("Rewards are not populated") {
            total += reward;
        }
        total
    }

    /// Numerical value that represent's how good the learning is.
    /// the closer to the 0 the better the learning
    fn get_learning_score_for_game(&mut self, n: usize) -> f64 {
        let data = self.for_game_df[n]
            .as_ref()
            .unwrap()
            .column("diff_actual_learned")
            .expect("Column not found");
        // let sum: f64 = data.abs().unwrap().sum().unwrap();
        let sum = data.sum().unwrap();  // Change to absolute sum 
        sum
    }

    fn display_statistics(&mut self) {
        println!("\n### Statistics for each dataframe separately. ###");

        for game_number in 0..self.game_runner.num_of_games {
            match self.for_game_df[game_number].clone() {
                Some(df) => {
                    println!("### Results Dataframe for trial : {} ###", game_number);
                    println!("{:?} \n", df);
                }
                None => {
                    println!("Dataframe for game: {} is not present.", game_number);
                }
            }
        }

        println!("### Statistics for all the games ###");
        println!(
            "Run {} games, with {} turns in each",
            NUM_OF_GAMES_TO_PLAY,
            NUM_OF_TURNS_IN_A_GAME
        );
        println!("Epsilon: {} \t Alpha: {}", EPSILON, ALPHA);
        println!("{:?}", self.df.as_ref().unwrap());
    }
}
