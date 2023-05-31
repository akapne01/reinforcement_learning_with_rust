/// Python version: https://www.dominodatalab.com/blog/k-armed-bandit-problem

use polars::prelude::*;

use crate::{
    bernulli_bandit::{ BernulliBandit, generate_random_number_in_range },
    simulation_runner::SimulationRunner,
    constants::{ PRINT_EACH_STEP, NUM_OF_BANDITS, NUM_OF_TRIALS },
};

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
    pub resulting_actions: Option<Vec<usize>>,
    pub resulting_rewards: Option<Vec<f64>>,
    pub df_results: Option<DataFrame>,
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
            resulting_actions: None,
            resulting_rewards: None,
            df_results: None,
        }
    }

    /// Runs game, populates statistics and prints them in console
    pub fn run_game(&mut self) {
        self.run_choose_random_action_all_the_time();
        self.calculate_statistics();
    }

    /// Runs all trials and records results.
    /// This function populates results vector that contains the bandit number selecte dand reward
    /// recieved on that trial.
    fn run_choose_random_action_all_the_time(&mut self) {
        let mut resulting_actions = vec![];
        let mut resulting_rewards = vec![];

        for trial in 0..self.no_of_trials as usize {
            let bandit_number = generate_random_number_in_range(
                0,
                (self.no_of_bandits - 1).try_into().unwrap()
            ) as usize;
            resulting_actions.insert(trial, bandit_number);
            resulting_rewards.insert(trial, self.bandits[bandit_number].pull());

            if PRINT_EACH_STEP {
                println!(
                    "Trial={} \t Playing bandit {} \t Reward is {}",
                    trial,
                    bandit_number,
                    resulting_rewards[trial]
                );
            }
        }
        self.resulting_actions = Some(resulting_actions);
        self.resulting_rewards = Some(resulting_rewards);
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
        if self.resulting_actions.is_none() || self.resulting_rewards.is_none() {
            self.run_choose_random_action_all_the_time();
        }
        std::env::set_var("POLARS_FMT_MAX_COLS", "12");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "12");
        let probabilities = self._get_actual_probabilities();

        let mut bandits_frequency = vec![0; self.no_of_bandits as usize];
        let mut bandits_rewards = vec![0.0; self.no_of_bandits as usize];

        for (&action, &reward) in self.resulting_actions
            .as_ref()
            .unwrap()
            .iter()
            .zip(self.resulting_rewards.as_ref().unwrap().iter()) {
            bandits_frequency[action] += 1;
            bandits_rewards[action] += reward;
        }

        let mut dfr = DataFrame::new(
            vec![
                Series::new("bandit", Vec::from_iter(0..self.no_of_bandits)),
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

    pub fn calculate_average_reward(&mut self) -> f64 {
        if self.resulting_rewards.is_none() {
            self.run_choose_random_action_all_the_time();
        }
        let total: f64 = self.resulting_rewards.as_ref().unwrap().iter().sum();
        total / (self.no_of_trials as f64)
    }
}

/// Public function used from main to run Multi-Armed Bernulli Bandits Game.
pub fn run() {
    let mut runner = SimulationRunner::new();
    runner.run_bernulli_multi_armed_simulation_game();
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

        assert!(game.resulting_actions.is_some(), "Should populate resulting actions vector");
        assert_eq!(
            game.resulting_actions.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "There is action recorded for each trial."
        );
        assert!(
            game.resulting_rewards.is_some(),
            "Should populate resulting rewards received vector"
        );
        assert_eq!(
            game.resulting_rewards.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "There is a reward recorded for each trial."
        );
    }

    #[test]
    fn test_calculate_statistics_when_no_results() {
        let mut game = BernulliMultiArmedBanditsGame::new();

        assert!(game.resulting_actions.is_none());
        assert!(game.resulting_rewards.is_none());
        assert!(game.df_results.is_none());

        game.calculate_statistics();

        assert!(
            game.resulting_actions.is_some(),
            "When actions taken don't exist they are populated."
        );
        assert!(game.resulting_rewards.is_some(), "When rewards don't exist they are populated.");
        assert_eq!(
            game.resulting_actions.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "For each trial action taken have been saved."
        );
        assert_eq!(
            game.resulting_rewards.unwrap().len(),
            NUM_OF_TRIALS as usize,
            "For each trial reward recieved has been saved."
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
