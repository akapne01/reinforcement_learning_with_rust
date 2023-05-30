/// Python version: https://www.dominodatalab.com/blog/k-armed-bandit-problem

use polars::prelude::*;

use crate::{
    bernulli_bandit::{ BernulliBandit, generate_random_number_in_range },
    simulation_runner::SimulationRunner,
};

use crate::constants::{ IS_VERBOSE_MODE, NUM_OF_BANDITS, NUM_OF_TRIALS };

/// This game represent's a solution to a simple the Multi-Armed bandit problem.
/// Here we have a Reinforcement Learning agent faced with multiple arm bandits,
/// it can be tought as agent having multiple slot machines in from of him.
/// At each step, agent chooses which leaver to pull.
/// After pulling a leaver, agent recieves a reward that is either: 0.0 or 1.0.
/// 0.0 represents loss, and 1.0 represents win.
/// Agent performs set number of trials and is faced with set number of leavers
/// to pull.
/// Agnet's objective is to maximize the total reward that it recieves over time
/// by learning the probabilities of each of winning when pressing each leaver.
#[derive(PartialEq, Debug)]
pub struct BernulliMultiArmedBanditsGame {
    no_of_bandits: u32,
    no_of_trials: u32,
    bandits: Vec<BernulliBandit>,
    results: Option<Vec<(usize, f64)>>,
    df_results: Option<DataFrame>,
}

impl BernulliMultiArmedBanditsGame {
    /// Initalize the game allowing to specify the number of multi-arm bandits
    /// and number of trials that you would like run.
    /// The bandits are created each with a random probability that is not known
    /// to the agent, but instead the agent is learning them over time.
    pub fn new() -> Self {
        BernulliMultiArmedBanditsGame {
            no_of_bandits: NUM_OF_BANDITS,
            no_of_trials: NUM_OF_TRIALS,
            bandits: BernulliBandit::new_as_vector(NUM_OF_BANDITS as usize),
            results: None,
            df_results: None,
        }
    }

    /// Runs game, populates statistics and prints them in console
    pub fn run_game(&mut self) {
        self.run_choose_random_action_all_the_time();
        self.calculate_statistics();
        self.print_statistics()
    }

    /// Runs all trials and records results.
    /// This function populates results vector that contains the bandit number selecte dand reward
    /// recieved on that trial.
    fn run_choose_random_action_all_the_time(&mut self) {
        let mut results: Vec<(usize, f64)> = vec![(0, 0.0); self.no_of_trials as usize];

        for trial in 0..self.no_of_trials as usize {
            let bandit_number = generate_random_number_in_range(
                0,
                (self.no_of_bandits - 1).try_into().unwrap()
            ) as usize;
            results[trial] = (bandit_number, self.bandits[bandit_number].pull());

            if IS_VERBOSE_MODE {
                println!(
                    "Trial={} \t Playing bandit {} \t Reward is {}",
                    trial,
                    results[trial].0,
                    results[trial].1
                );
            }
        }
        self.results = Some(results);
    }

    /// Helper method that allows to obtain the actula probabilities that each arm bandit had
    /// for statistical purposes only.
    /// The mean rewards coverge over time and with many trials to the actual proababilities reflecting
    /// that playing agent has learned them.
    fn _get_actual_probabilities(&self) -> Vec<f64> {
        let mut probabilities = vec![0.0; self.no_of_bandits as usize];
        for (index, bandit) in self.bandits.iter().enumerate() {
            probabilities[index] = bandit.get_probablity();
        }
        probabilities
    }

    /// This method writes results in the dataframe and adds additional calculations.
    /// If results vector is not present, then will run to create it.
    fn calculate_statistics(&mut self) {
        if self.results.is_none() {
            self.run_choose_random_action_all_the_time();
        }
        std::env::set_var("POLARS_FMT_MAX_COLS", "12");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "12");
        let probabilities = self._get_actual_probabilities();
        let mut bandits = vec![0; self.no_of_bandits as usize];
        let mut bandits_frequency = vec![0; self.no_of_bandits as usize];
        let mut bandits_rewards = vec![0.0; self.no_of_bandits as usize];
        for &(bandit, reward) in self.results.as_ref().unwrap().into_iter() {
            bandits[bandit] = bandit as i32;
            bandits_frequency[bandit] += 1;
            bandits_rewards[bandit] += reward;
        }

        let mut dfr = DataFrame::new(
            vec![
                Series::new("bandit", &bandits),
                Series::new("actual_probability", &probabilities),
                Series::new("frequency", &bandits_frequency),
                Series::new("total_reward", &bandits_rewards)
            ]
        ).expect("Failed to create DataFrame");

        dfr = dfr
            .lazy()
            .with_column((col("total_reward") / col("frequency")).alias("mean_reward"))
            .collect()
            .unwrap();

        dfr = dfr
            .lazy()
            .with_column((col("actual_probability") / col("frequency")).alias("diff"))
            .collect()
            .unwrap();

        dfr = dfr.sort(["total_reward"], true).expect("Couldn't sort the dataframe");
        self.df_results = Some(dfr);
    }

    /// This method prints in console polars dataframe that holds calculated statistics.
    /// If statistics is not populated, then will calculate them and generate dataframe.
    fn print_statistics(&mut self) {
        if self.df_results.is_none() {
            self.calculate_statistics();
        }
        println!("## Multi Armed Bandit Statistics ##");
        let df = self.df_results.as_ref().unwrap();
        println!("{:?}", df);
    }
}

/// Public function used from main to run Multi-Armed Bernulli Bandits Game.
pub fn run() {
    let mut runner = SimulationRunner::new_bernulli_bandit();
    runner.game.run_game();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_creation_of_game() {
        let game = BernulliMultiArmedBanditsGame::new();

        assert_eq!(game.no_of_bandits, NUM_OF_BANDITS);
        assert_eq!(game.no_of_trials, NUM_OF_TRIALS);
        assert_eq!(game.bandits.is_empty(), false);
        assert_eq!(game.bandits.len(), 10);
    }

    #[test]
    fn test_getting_actual_probabilities() {
        let game = BernulliMultiArmedBanditsGame::new();

        let probabilities = game._get_actual_probabilities();

        assert_eq!(probabilities.is_empty(), false);
        assert_eq!(probabilities.len(), NUM_OF_BANDITS as usize);

        for &probability in &probabilities {
            assert!(probability >= 0.0);
            assert!(probability <= 1.0, "Proability is between 0 and 1");
        }
    }

    #[test]
    fn test_running_game_stohastically() {
        let mut game = BernulliMultiArmedBanditsGame::new();

        game.run_choose_random_action_all_the_time();

        assert!(game.results.is_some(), "Should populate results vector");
        assert_eq!(
            game.results.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "There is a result for each trial."
        );
    }

    #[test]
    fn test_calculate_statistics_when_no_results() {
        let mut game = BernulliMultiArmedBanditsGame::new();

        assert!(game.results.is_none());
        assert!(game.df_results.is_none());

        game.calculate_statistics();

        assert!(game.results.is_some(), "When results don't exist they are populated.");
        assert_eq!(
            game.results.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "Each trial has result saved."
        );
        assert!(game.df_results.is_some(), "Dataframe with results is populated.");
        assert_eq!(game.df_results.as_ref().unwrap().is_empty(), false, "Dataframe has data.");
        assert_eq!(
            game.df_results.unwrap().shape().0,
            NUM_OF_BANDITS as usize,
            "Row in the dataframe exist for each representing each bandit."
        );
    }
}
