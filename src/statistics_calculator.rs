use std::{ path::Path, fs::{ self, File } };

use polars::prelude::*;
use chrono::prelude::*;
use std::io::prelude::*;
use std::cmp;

use crate::{
    bernoulli_multi_armed_bandits_game::BernoulliParallelGameRunner,
    constants::{ NUM_OF_GAMES_TO_PLAY, NUM_OF_TURNS_IN_A_GAME, EPSILON, ALPHA, NUM_OF_BANDITS },
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
        self.save_per_game_df();
        self.save_summary_df_for_all_games();
        self.write_statistics();
    }

    /// Populates dataframe for each game separately
    fn save_per_game_df(&mut self) {
        for n in 0..self.game_runner.num_of_games {
            self.populate_dataframe_for_game(n);
        }
    }

    fn save_summary_df_for_all_games(&mut self) {
        // rows: game
        // columns: mean_reward, total_reward, correctness_score
        let mut means = Vec::new();
        let mut scores = Vec::new();
        let mut totals = Vec::new();

        (0..self.game_runner.num_of_games).for_each(|n: usize| {
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

        if game.resulting_actions.is_none() || game.resulting_rewards.is_none() {
            self.game_runner.run_all_games_in_parallel();
            return self.populate_dataframe_for_game(n);
        }

        let num_of_bandits = game.num_of_bandits;

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
            return self.get_mean_reward_per_game(n);
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
            return self.get_total_reward_per_game(n);
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
        if self.for_game_df[n].is_none() {
            self.save_per_game_df();
            return self.get_learning_score_for_game(n);
        }

        let data = self.for_game_df[n]
            .as_ref()
            .unwrap()
            .column("diff_actual_learned")
            .expect("Column not found");
        let abs_series: Vec<f64> = data
            .f64()
            .unwrap()
            .into_iter()
            .map(|n| n.unwrap_or(0.0).abs())
            .collect();
        abs_series.into_iter().sum()
    }

    fn write_statistics(&mut self) {
        // Set environment variables so that the whole dataframe is printed when written to the file.
        let max_rows = cmp::max(NUM_OF_GAMES_TO_PLAY, NUM_OF_BANDITS);
        std::env::set_var("POLARS_FMT_MAX_COLS", "12");
        std::env::set_var("POLARS_FMT_MAX_ROWS", max_rows.to_string());

        let base_directory = "files";
        let sub_directory = "bernoulli_multi_armed_bandits";
        let local: DateTime<Local> = Local::now();
        let datetime_str: &str = &local.format("%Y-%m-%d_%H:%M:%S").to_string();
        // let file_name = "run_result.txt";
        let file_name = format!("run_result_{}.txt", datetime_str);

        let base_directory_path = Path::new(base_directory);
        if !base_directory_path.is_dir() {
            // Create the base directory
            match fs::create_dir(base_directory_path) {
                Ok(_) => println!("Base directory '{}' created successfully", base_directory),
                Err(err) => {
                    eprintln!("Failed to create base directory: {}", err);
                    return;
                }
            }
        }

        // Check if the sub directory exists
        let sub_directory_path = base_directory_path.join(sub_directory);
        if sub_directory_path.is_dir() {
            println!("Sub directory '{}' already exists", sub_directory);
        } else {
            // Create the sub directory
            match fs::create_dir(&sub_directory_path) {
                Ok(_) => println!("Sub directory '{}' created successfully", sub_directory),
                Err(err) => eprintln!("Failed to create sub directory: {}", err),
            }
        }

        // Create the file path
        let file_path = sub_directory_path.join(file_name);
        let mut output = File::create(file_path).unwrap();

        // Check if required data exists, if it does not, then populate it
        if self.df.is_none() {
            self.save_summary_df_for_all_games();
            return self.write_statistics();
        }
        for n in 0..self.game_runner.num_of_games {
            if self.for_game_df[n].is_none() {
                self.save_per_game_df();
                return self.write_statistics();
            }
        }

        // Writing to the file
        writeln!(output, "\n### Statistics for each dataframe separately. ###").unwrap();

        for game_number in 0..self.game_runner.num_of_games {
            match self.for_game_df[game_number].clone() {
                Some(df) => {
                    writeln!(
                        output,
                        "### Results Dataframe for trial : {} ###",
                        game_number
                    ).unwrap();
                    writeln!(output, "{:?} \n", df).unwrap();
                }
                None => {
                    writeln!(
                        output,
                        "Dataframe for game: {} is not present.",
                        game_number
                    ).unwrap();
                }
            }
        }

        writeln!(output, "### Statistics for all the games ###").unwrap();
        writeln!(
            output,
            "Run {} games, with {} turns in each",
            NUM_OF_GAMES_TO_PLAY,
            NUM_OF_TURNS_IN_A_GAME
        ).unwrap();
        writeln!(output, "Epsilon: {} \t Alpha: {}", EPSILON, ALPHA).unwrap();
        writeln!(output, "{:?}", self.df.as_ref().unwrap()).unwrap();
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_creation_of_new_agent_game_statistics_wrapper() {
        let stats = BernoulliAgentStatisticsWrapper::new();

        assert_eq!(stats.game_runner.num_of_games, NUM_OF_GAMES_TO_PLAY);
        assert_eq!(stats.game_runner.games.len(), NUM_OF_GAMES_TO_PLAY);
        assert_eq!(stats.for_game_df, vec![None; NUM_OF_GAMES_TO_PLAY]);
        assert_eq!(stats.df, None);
    }

    #[test]
    fn test_creation_of_statistics_wrapper_from_parallel_game_runner() {
        let runner = BernoulliParallelGameRunner::new();
        let stats = BernoulliAgentStatisticsWrapper::from(runner);

        assert_eq!(stats.game_runner.num_of_games, NUM_OF_GAMES_TO_PLAY);
        assert_eq!(stats.game_runner.games.len(), NUM_OF_GAMES_TO_PLAY);
        assert_eq!(stats.for_game_df, vec![None; NUM_OF_GAMES_TO_PLAY]);
        assert_eq!(stats.df, None);
    }

    #[test]
    fn test_mean_calculation_for_game_when_results_recorded() {
        let game_number: usize = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        stats.game_runner.games[game_number].run_one_game();

        let mean_reward = stats.get_mean_reward_per_game(game_number);

        assert!(stats.game_runner.games[game_number].resulting_rewards.is_some());
        assert!((0.0..1.0).contains(&mean_reward));
        assert_relative_eq!(mean_reward, 0.5, epsilon = 0.49);
    }

    #[test]
    fn test_mean_calculation_for_game_when_no_rewards_recorded() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();

        assert!(stats.game_runner.games[n].resulting_rewards.is_none());

        let result = stats.get_mean_reward_per_game(n);

        assert!(stats.game_runner.games[n].resulting_rewards.is_some());
        assert!((0.0..1.0).contains(&result));
        assert_relative_eq!(result, 0.5, epsilon = 0.49);
    }

    #[test]
    fn test_total_rewards_per_game_when_no_rewards_recorded() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        let max_value = stats.game_runner.games[0].num_of_turns as f64;
        assert!(stats.game_runner.games[n].resulting_rewards.is_none());

        let result = stats.get_total_reward_per_game(n);
        println!("{}", result);
        assert!(stats.game_runner.games[n].resulting_rewards.is_some());
        assert!((0.0..max_value).contains(&result));
    }

    #[test]
    fn test_total_rewards_per_game_when_rewards_are_recorded() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        let max_value = stats.game_runner.games[0].num_of_turns as f64;
        stats.game_runner.run_all_games_in_parallel();

        let result = stats.get_total_reward_per_game(n);

        assert!(stats.game_runner.games[n].resulting_rewards.is_some());
        assert!((0.0..max_value).contains(&result));
    }

    #[test]
    fn test_populate_df_for_game_when_actions_and_rewards_not_recorded() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        assert_eq!(stats.for_game_df, vec![None; stats.game_runner.num_of_games]);
        assert!(stats.game_runner.games[n].resulting_actions.is_none());
        assert!(stats.game_runner.games[n].resulting_rewards.is_none());

        stats.populate_dataframe_for_game(n);

        assert!(stats.for_game_df[n].is_some());
        assert_eq!(
            stats.for_game_df[n].as_ref().unwrap().shape().0,
            stats.game_runner.games[n].num_of_bandits
        );
    }

    #[test]
    fn test_populate_df_for_game_when_actions_and_rewards_are_recorded() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        stats.game_runner.games[n].run_one_game();

        assert!(stats.game_runner.games[n].resulting_actions.is_some());
        assert!(stats.game_runner.games[n].resulting_rewards.is_some());

        stats.populate_dataframe_for_game(n);

        assert!(stats.for_game_df[n].is_some());
        assert_eq!(
            stats.for_game_df[n].as_ref().unwrap().shape().0,
            stats.game_runner.games[n].num_of_bandits
        );
    }

    #[test]
    fn test_saving_dataframe_for_each_game_separately() {
        let mut stats = BernoulliAgentStatisticsWrapper::new();

        stats.save_per_game_df();

        for n in 0..stats.game_runner.num_of_games {
            assert!(stats.for_game_df[n].is_some());
        }
    }

    #[test]
    fn test_get_learning_score_per_game_when_per_game_df_present() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();
        stats.save_per_game_df();

        assert!(stats.for_game_df[n].is_some());

        let result = stats.get_learning_score_for_game(n);

        assert!(result >= 0.0);
    }

    #[test]
    fn test_get_learning_score_per_game_when_per_game_df_not_populated() {
        let n = 0;
        let mut stats = BernoulliAgentStatisticsWrapper::new();

        assert!(stats.for_game_df[n].is_none());

        let result = stats.get_learning_score_for_game(n);

        assert!(stats.for_game_df[n].is_some());
        assert!(result > 0.0);
    }

    #[test]
    fn test_save_df_summary_for_all_games() {
        let mut stats = BernoulliAgentStatisticsWrapper::new();

        stats.save_summary_df_for_all_games();

        assert!(stats.df.is_some());
        assert_eq!(stats.df.unwrap().shape().0, stats.game_runner.num_of_games);
    }

    #[test]
    fn test_get_actual_probabilities() {
        let n = 0;
        let stats = BernoulliAgentStatisticsWrapper::new();

        let probabilities = stats._get_actual_probabilities_per_game(n);

        assert_eq!(probabilities.is_empty(), false);
        assert_eq!(probabilities.len(), stats.game_runner.games[n].num_of_bandits);
    }

    #[test]
    fn test_run_method() {
        let mut stats = BernoulliAgentStatisticsWrapper::new();

        stats.run();

        // Test that all games where run, and we have a vector of all actions taken and rewards recieved
        for n in 0..stats.game_runner.num_of_games {
            assert!(stats.game_runner.games[n].resulting_actions.is_some());
            assert!(stats.game_runner.games[n].resulting_rewards.is_some());
        }

        // Datafarame is populated for each game separately
        for df in stats.for_game_df {
            assert!(df.is_some());
            assert_eq!(df.unwrap().shape().0, stats.game_runner.num_of_games);
        }

        // Dataframe is populated that summarizes all the games
        assert!(stats.df.is_some());
        assert_eq!(stats.df.unwrap().shape().0, stats.game_runner.num_of_games);
    }
}
