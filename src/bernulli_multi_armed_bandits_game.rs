use polars::prelude::*;

use crate::{
    bernulli_bandit::{
        BernulliBandit,
        generate_uniform_random_number,
        generate_random_number_in_range,
    },
    constants::{
        PRINT_EACH_STEP,
        NUM_OF_BANDITS,
        NUM_OF_TURNS_IN_A_GAME,
        EPSILON,
        ALPHA,
        IS_VERBOSE_MODE,
    },
};

/// This game represent's a solution to a simple the Multi-Armed bandit problem.
/// Here we have a Reinforcement Learning agent faced with multiple arm bandits,
/// it can be tought as agent having multiple slot machines in from of him.
/// At each step, agent chooses which leaver to pull.
/// After pulling a leaver, agent recieves a reward that is either: 0.0 or 1.0.
/// 0.0 represents loss, and 1.0 represents win.
/// Agent performs set number of trials and is faced with set number of leavers
/// to pull.
/// Agent's objective is to maximize the total reward that it recieves over time
/// by learning the probabilities of each of winning when pressing each leaver.
#[derive(PartialEq, Debug)]
pub struct BernulliMultiArmedBanditsGame {
    /// Number of slot machines to play in one game.
    num_of_bandits: usize,
    /// Represents the number of turns taken in each game.
    num_of_turns: usize,
    /// This vector holds the actual bandits that are genearted each
    /// with the random probability that agent does not know, but is
    /// attempting to learn
    bandits: Vec<BernulliBandit>,
    /// This vector represents the long term knowledge that RL agent has learned.
    /// It corresponds to the value function. Index of this vector represents the
    /// number of the armed bandit. Value recorded in the vector is an estimation
    /// of the probability of winning. The values recorded in this vector represents
    /// the knowledge that RL agent has.
    q_values: Vec<f64>,
    /// Should be in range: 0 <= epsilon <= 1
    /// Represents the proabability with which to take exploratory action (random action)
    /// If epsilon = 0, greedy action is always taken. If epsilon = 1, random action is
    /// always taken.
    epsilon: f64,
    /// Stepsize parameter used when updating value function. Determines how much
    /// weight to give to the rewards recieved from newly discovered actions.
    alpha: f64,
    /// Records action taken on each turn. When none, means that the
    /// game hasn't been played yet. Index represents the turn in the
    /// game.
    pub resulting_actions: Option<Vec<usize>>,
    /// Records rewards that resuled from taking each action. When none,
    /// the game hasn't been played. Index represents the turn in the game.
    pub resulting_rewards: Option<Vec<f64>>,
    /// Summaraizes the results in a dataframe for each armed bandit.
    /// Records total reward recieved in each game, and how many times each
    /// of the slots where selected to be pulled.
    /// For comparison purposes also keeps a record of the actual probability
    /// of winning that was set for each of the slot machines to win.
    pub df_results: Option<DataFrame>,
}

impl BernulliMultiArmedBanditsGame {
    /// Initalize the game allowing to specify the number of multi-arm bandits
    /// and number of trials that you would like run.
    /// The bandits are created each with a random probability that is not known
    /// to the agent, but instead the agent is learning them over time.
    pub fn new() -> Self {
        BernulliMultiArmedBanditsGame {
            num_of_bandits: NUM_OF_BANDITS,
            num_of_turns: NUM_OF_TURNS_IN_A_GAME,
            bandits: BernulliBandit::new_as_vector(NUM_OF_BANDITS as usize),
            q_values: vec![0.0; NUM_OF_BANDITS as usize],
            epsilon: EPSILON,
            alpha: ALPHA,
            resulting_actions: None,
            resulting_rewards: None,
            df_results: None,
        }
    }

    /// Runs game, populates statistics and prints them in console
    pub fn run_game(&mut self) {
        self.run_and_record_resuts();
        self.calculate_statistics();
    }

    /// This action selection policy chooses a random action every single time
    /// This is action policy that is always explorative as agent never exploits
    /// the knowledge that it has learned.
    pub fn policy_select_action_randomly(&self) -> usize {
        let action = generate_random_number_in_range(0, self.num_of_bandits) as usize;
        if PRINT_EACH_STEP {
            println!("# Random Action selected :{} #", action);
        }
        action
    }

    /// If epsilon-greedy action policy is selected, this method updates the agent
    /// estiamted Q values which in this case represent the proability of winning
    /// for each of the armed bandit (slot machine).
    /// new_estimate = old_estimate + step_size (target - old_estimate)
    fn update_value_function(&mut self, action: usize, reward: f64) {
        self.q_values[action] += self.alpha * (reward - self.q_values[action]);
        if PRINT_EACH_STEP {
            println!(
                "# Updating value function. Alpha: {} \t action: {} \t reward: {} #",
                self.alpha,
                action,
                reward
            );
            println!("   After update: {:?}", self.q_values);
        }
    }

    /// Action selection policy: epsilon greedy.
    /// Epsilon bounds: 0 <= epsilon <= 1.
    /// This action selection policy balances the exporation and exploitation.
    /// The actions are selected greedily with probability of (1 - epsilon).
    /// When actions are selected greedily, the RL agent exploits what it has learned.
    /// Actions are selected randomly with probability of epsilon. When actions are
    /// selected randomly, agent explores the action space to gain a new knowledge
    /// or refine the knowledge that it already has.
    pub fn policy_select_epsilon_greedy_action(&self) -> usize {
        let number = generate_uniform_random_number();
        if number <= self.epsilon {
            return self.policy_select_action_randomly();
        }
        let max_value = self.q_values.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
        let max_indices: Vec<usize> = self.q_values
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == max_value)
            .map(|(index, _)| index)
            .collect();

        if PRINT_EACH_STEP {
            println!("# Selecting Action with epsilon greedy policy. Epsilon: {} #", self.epsilon);
            println!("Max Q value: {}", max_value);
            println!("Indices that correspond to this value: {:?}", max_indices);
        }

        let random_index = generate_random_number_in_range(0, max_indices.len());

        if PRINT_EACH_STEP {
            println!("# Greedy Action selected :{} #", max_indices[random_index]);
        }
        max_indices[random_index]
    }

    /// Runs all trials and records results.
    /// This function populates results vector that contains the bandit number selecte dand reward
    /// recieved on that trial.
    fn run_and_record_resuts(&mut self) {
        // const NUM_EXPLORATORY_TRIALS: usize = 1000;
        // const MIN_EPSILON: f64 = 0.05;
        let mut resulting_actions = vec![];
        let mut resulting_rewards = vec![];

        for turn in 0..self.num_of_turns as usize {
            // For first n actions only explore. Then exploratory
            // action is taken with probability of MIN_EPSILON.
            // if turn == NUM_EXPLORATORY_TRIALS {
            //     self.set_epsilon(MIN_EPSILON);
            // }
            let action_to_take = self.policy_select_epsilon_greedy_action();
            let reward = self.bandits[action_to_take].pull();

            // Step size parameter alpha changes at every step and is
            // 1/n which corresponds to the average mean.
            // The first trials are given bigger weight and importance
            // and as n increases the importance given to the recieved
            // reward decreases.
            // let alpha = 1.0 / ((turn as f64) + 1.0);
            // self.set_alpha(alpha);

            self.update_value_function(action_to_take, reward);
            resulting_actions.insert(turn, action_to_take);
            resulting_rewards.insert(turn, reward);

            if PRINT_EACH_STEP {
                println!(
                    "\nTurn={} \t Playing bandit {} \t Reward is {}",
                    turn,
                    action_to_take,
                    resulting_rewards[turn]
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
        let mut probabilities = vec![0.0; self.num_of_bandits as usize];
        for (index, bandit) in self.bandits.iter().enumerate() {
            probabilities[index] = bandit.get_probablity();
        }
        probabilities
    }

    /// This method writes results in the dataframe and adds additional calculations.
    /// If results vector is not present, then will run to create it.
    fn calculate_statistics(&mut self) {
        if self.resulting_actions.is_none() || self.resulting_rewards.is_none() {
            self.run_and_record_resuts();
        }
        std::env::set_var("POLARS_FMT_MAX_COLS", "12");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "12");
        let probabilities = self._get_actual_probabilities();

        let mut bandits_frequency = vec![0; self.num_of_bandits as usize];
        let mut bandits_rewards = vec![0.0; self.num_of_bandits as usize];

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
                Series::new("bandit", Vec::from_iter(0..self.num_of_bandits as u32)),
                Series::new("actual_probability", &probabilities),
                Series::new("learned_probability", &self.q_values),
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
            .with_column((col("actual_probability") - col("mean_reward")).alias("diff_actual_mean"))
            .collect()
            .unwrap();

        dfr = dfr
            .lazy()
            .with_column(
                (col("actual_probability") - col("learned_probability")).alias(
                    "diff_actual_learned"
                )
            )
            .collect()
            .unwrap();

        dfr = dfr
            .lazy()
            .with_column(
                (col("mean_reward") - col("learned_probability")).alias("diff_mean_learned")
            )
            .collect()
            .unwrap();

        dfr = dfr.sort(["actual_probability"], true).expect("Couldn't sort the dataframe");
        self.df_results = Some(dfr);
    }

    /// Calculates the average reward received in each turn
    pub fn calculate_mean_reward(&mut self) -> f64 {
        if self.resulting_rewards.is_none() {
            self.run_and_record_resuts();
        }
        let total: f64 = self.resulting_rewards.as_ref().unwrap().iter().sum();
        total / (self.num_of_turns as f64)
    }

    /// Allows to change epsilon value for differnt games
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Allows to change alpha value for differnt games
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Calculates the total rewards received by playing game
    pub fn calculate_total_reward(&mut self) -> f64 {
        if self.resulting_rewards.is_none() {
            self.run_and_record_resuts();
        }
        let mut total = 0.0;
        for reward in self.resulting_rewards.as_ref().expect("Rewards are not populated") {
            total += reward;
        }
        total
    }

    /// Numerical value that represent's how good the learning is.
    /// the closer to the 0 the better the score.
    pub fn calculate_learning_correctness_score(&mut self) -> f64 {
        // from df take 2 columns: actual_probability and learned_probability
        // we have a column: diff_actual_learned. We want to take abs and sum all the elements
        let data = self.df_results
            .as_ref()
            .unwrap()
            .column("diff_actual_learned")
            .expect("Column not found");
        let sum: f64 = data.abs().unwrap().sum().unwrap();
        sum
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_creation_of_game() {
        let game = BernulliMultiArmedBanditsGame::new();

        assert_eq!(game.num_of_bandits, NUM_OF_BANDITS);
        assert_eq!(game.num_of_turns, NUM_OF_TURNS_IN_A_GAME);
        assert_eq!(game.bandits.is_empty(), false);
        assert_eq!(game.bandits.len(), NUM_OF_BANDITS);
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

        game.run_and_record_resuts();

        assert!(game.resulting_actions.is_some(), "Should populate resulting actions vector");
        assert_eq!(
            game.resulting_actions.unwrap().len(),
            NUM_OF_TURNS_IN_A_GAME as usize,
            "There is action recorded for each trial."
        );
        assert!(
            game.resulting_rewards.is_some(),
            "Should populate resulting rewards received vector"
        );
        assert_eq!(
            game.resulting_rewards.unwrap().len(),
            NUM_OF_TURNS_IN_A_GAME as usize,
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
            NUM_OF_TURNS_IN_A_GAME as usize,
            "For each trial action taken have been saved."
        );
        assert_eq!(
            game.resulting_rewards.unwrap().len(),
            NUM_OF_TURNS_IN_A_GAME as usize,
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

    #[test]
    fn test_random_action_within_the_range_returned() {
        let game = BernulliMultiArmedBanditsGame::new();
        let turns = 10_000;
        let expected_mean = (0.0 + (NUM_OF_BANDITS as f64) - 1.0) / 2.0;
        let mut actions_taken = vec![];
        for i in 0..turns {
            // Always select random action
            let action = game.policy_select_action_randomly();
            assert!(action < (game.num_of_bandits as u32).try_into().unwrap());
            actions_taken.insert(i, action);
        }
        let sum: usize = actions_taken.iter().sum();
        let actual_mean = (sum as f64) / (turns as f64);

        let expected_range = expected_mean - 0.5..expected_mean + 0.5;
        assert!(
            expected_range.contains(&actual_mean),
            "We are working with probabilities and actual mean can vary a bit from expected mean so checking if it's in the range."
        );
    }

    #[test]
    fn test_update_value_function() {
        let mut game = BernulliMultiArmedBanditsGame::new();
        // Only update value function if exploratory action taken?
        let mut expected_value_function = vec![0.0; NUM_OF_BANDITS as usize];

        // 1st update
        assert_eq!(game.q_values, expected_value_function);
        game.update_value_function(0, 1.0);
        expected_value_function[0] = game.alpha * (1.0 - 0.0);
        assert_eq!(game.q_values, expected_value_function);

        println!("After 1st turn the expected value function: {:?}", expected_value_function);
        println!("After 1st turn the actual value function: {:?}", game.q_values);

        // 2nd update
        game.update_value_function(0, 1.0);
        expected_value_function[0] += game.alpha * (1.0 - expected_value_function[0]);
        assert_eq!(game.q_values, expected_value_function);

        println!("After 3rd turn the expected value function: {:?}", expected_value_function);
        println!("After 3rd turn the actual value function: {:?}", game.q_values);
    }

    #[test]
    fn test_epsilon_greedy_action_selection_policy() {
        let turns = 100_000;

        // Initalize game and set learned Q values to reflect the  actual probabilities for testing purposes
        let mut game = BernulliMultiArmedBanditsGame::new();
        game.q_values = game._get_actual_probabilities();

        // Calculate actual Maximum probability
        let max_value = game
            ._get_actual_probabilities()
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, f64::max);

        // Calculate to which index the maximum probability corresponds to
        // This index denotes a action that should be taken the most often.
        let max_indices: Vec<usize> = game
            ._get_actual_probabilities()
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == max_value)
            .map(|(index, _)| index)
            .collect();

        // Simulate playing the game and record actions that are returned to be
        // taken on each turn.
        let mut actions_taken = vec![];
        for i in 0..turns {
            let action = game.policy_select_epsilon_greedy_action();
            assert!(
                action < (game.num_of_bandits as u32).try_into().unwrap(),
                "Don't expect action number to be bigger than number of slots"
            );
            actions_taken.insert(i, action);
        }

        // Calculate the expected min times greedy action was taken based on a given EPSILON.
        let num_times_greedy_action_expected = (1.0 - EPSILON) * (turns as f64);

        // Count how many times greedy action was actually taken.
        let num_times_greedy_action_actually_taken = actions_taken
            .iter()
            .filter(|&value| *value == max_indices[0])
            .count() as f64;

        if IS_VERBOSE_MODE {
            println!("\n# Actual max value: {}", max_value);
            println!("# The actual max index: {:?}", max_indices);
            println!("# Actual actions taken: {:?}", actions_taken);
            println!(
                "# Minimum Number of times greedy action expected: {:?}",
                num_times_greedy_action_expected
            );
            println!(
                "# Number of times greedy action actually taken: {:?}",
                num_times_greedy_action_actually_taken
            );
        }

        assert!(
            num_times_greedy_action_actually_taken >= num_times_greedy_action_expected,
            "Assert that greedy action was taken more or equal number of times than expected with probability (1 - epsilon)."
        );
    }
}
