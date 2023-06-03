use crate::bernoulli_bandit::{
    BernoulliBanditEnvironment,
    generate_random_number_in_range,
    generate_uniform_random_number,
};
use crate::constants::{
    PRINT_EACH_STEP,
    NUM_OF_BANDITS,
    NUM_OF_TURNS_IN_A_GAME,
    ALPHA,
    EPSILON,
    USE_AVERAGE_VALUE_UPDATE,
    SHOULD_DECKAY_EPSILON,
    NUM_OF_GAMES_TO_PLAY,
};

/// This game represents a solution to the simple Multi-Armed Bandit problem.
/// The agent is faced with multiple slot machines (bandits) and must learn
/// the probabilities of winning for each bandit to maximize the total reward.
/// This struct is responsible for running one game and recording action taken
/// and rewards received for it in resulting_actions and resulting_rewards
/// vectors.
#[derive(PartialEq, Debug, Clone)]
pub struct BernoulliOneGameLearningAgent {
    /// Number of slot machines to play in one game.
    pub num_of_bandits: usize,
    /// Represents the number of turns taken in each game.
    pub num_of_turns: usize,
    /// This vector holds the actual bandits that are genearted each
    /// with the random probability that agent does not know, but is
    /// attempting to learn
    pub environment: Vec<BernoulliBanditEnvironment>,
    /// This vector represents what RL agent has learned over time about each bandit.
    /// It corresponds to the value function. Index of this vector represents the
    /// number of the armed bandit. Value recorded in the vector is an estimation
    /// of the probability of winning.
    pub q_values: Vec<f64>,
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
    // pub df_results: Option<DataFrame>,
    /// Contains total reward and number of times the action was taken
    total_reward_per_bandit: Vec<f64>,
    /// Analogus to using alpha parameter. Initalized to 0. Represent number
    /// of times each slot (bandit) was selected.
    num_times_bandit_selected: Vec<usize>,
}

impl BernoulliOneGameLearningAgent {
    /// Initalize the game allowing to specify the number of multi-arm bandits
    /// and number of trials that you would like run.
    /// The bandits are created each with a random probability that is not known
    /// to the agent, but instead the agent is learning them over time.
    pub fn new() -> Self {
        BernoulliOneGameLearningAgent {
            num_of_bandits: NUM_OF_BANDITS,
            num_of_turns: NUM_OF_TURNS_IN_A_GAME,
            environment: BernoulliBanditEnvironment::new_as_vector(NUM_OF_BANDITS),
            q_values: vec![0.0; NUM_OF_BANDITS as usize],
            total_reward_per_bandit: vec![0.0; NUM_OF_BANDITS],
            num_times_bandit_selected: vec![0; NUM_OF_BANDITS],
            epsilon: EPSILON,
            alpha: ALPHA,
            resulting_actions: None,
            resulting_rewards: None,
        }
    }

    pub fn run_one_game(&mut self) {
        const NUM_EXPLORATORY_TRIALS: usize = 1000;
        const MIN_EPSILON: f64 = 0.01;
        let mut resulting_actions = vec![];
        let mut resulting_rewards = vec![];

        for turn in 0..self.num_of_turns as usize {
            if SHOULD_DECKAY_EPSILON {
                if turn == NUM_EXPLORATORY_TRIALS {
                    self.epsilon = MIN_EPSILON;
                }
            }
            let action_taken = self.epsilon_greedy_action_selection_policy();
            let reward = self.environment[action_taken].step();

            if USE_AVERAGE_VALUE_UPDATE {
                self.update_average_value_function(action_taken, reward);
            } else {
                self.update_value_function(action_taken, reward);
            }

            resulting_actions.insert(turn, action_taken);
            resulting_rewards.insert(turn, reward);

            if PRINT_EACH_STEP {
                println!(
                    "\nTurn={} \t Playing bandit {} \t Reward is {}",
                    turn,
                    action_taken,
                    resulting_rewards[turn]
                );
            }
        }
        self.resulting_actions = Some(resulting_actions);
        self.resulting_rewards = Some(resulting_rewards);
    }

    fn epsilon_greedy_action_selection_policy(&mut self) -> usize {
        assert!((0.0..=1.0).contains(&self.epsilon), "Epsilon must be in the range [0, 1].");
        let number = generate_uniform_random_number();
        if number <= self.epsilon {
            return self.random_action_selection_policy();
        }
        return self.greedy_action_selection_policy();
    }

    fn random_action_selection_policy(&self) -> usize {
        let action = generate_random_number_in_range(0, self.num_of_bandits) as usize;

        if PRINT_EACH_STEP {
            println!("# Random Action selected :{} #", action);
        }

        action
    }

    fn greedy_action_selection_policy(&mut self) -> usize {
        let max_value = self.q_values.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
        let max_indices: Vec<usize> = self.q_values
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == max_value)
            .map(|(index, _)| index)
            .collect();

        let random_index = generate_random_number_in_range(0, max_indices.len());

        if PRINT_EACH_STEP {
            println!("# Selecting Action with epsilon greedy policy. Epsilon: {} #", self.epsilon);
            println!("# Max Q value: {} #", max_value);
            println!("# Indices that correspond to this value: {:?} #", max_indices);
            println!("# Greedy Action selected :{} #", max_indices[random_index]);
        }

        max_indices[random_index]
    }

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

    /// Simulates mean average update for each bandit
    fn update_average_value_function(&mut self, action: usize, reward: f64) {
        self.num_times_bandit_selected[action] += 1;
        self.total_reward_per_bandit[action] += reward;

        let alpha = 1.0 / (self.num_times_bandit_selected[action] as f64);
        self.q_values[action] += alpha * (reward - self.q_values[action]);

        if PRINT_EACH_STEP {
            println!(
                "# Updating value function. Alpha: {} \t action: {} \t reward: {} #",
                alpha,
                action,
                reward
            );
            println!("   After update: {:?}", self.q_values);
        }
    }
}

/// This struct allows to run multiple Bernoulli Multiple Armed Bandit games
pub struct BernoulliGameLearningRunner {
    pub num_of_games: usize,
    pub games: Vec<BernoulliOneGameLearningAgent>,
}

impl BernoulliGameLearningRunner {
    pub fn new() -> Self {
        let games: Vec<BernoulliOneGameLearningAgent> = (0..NUM_OF_GAMES_TO_PLAY)
            .map(|_| BernoulliOneGameLearningAgent::new())
            .collect();

        BernoulliGameLearningRunner {
            num_of_games: NUM_OF_GAMES_TO_PLAY,
            games,
        }
    }

    pub fn run_all_games(&mut self) {
        for n in 0..self.num_of_games as usize {
            self.games[n].run_one_game();
        }
    }
}

#[cfg(test)]
mod test {
    use crate::constants::IS_VERBOSE_MODE;

    use super::*;

    #[test]
    fn test_creation_of_game() {
        let game = BernoulliOneGameLearningAgent::new();

        assert_eq!(game.num_of_bandits, NUM_OF_BANDITS);
        assert_eq!(game.num_of_turns, NUM_OF_TURNS_IN_A_GAME);
        assert_eq!(game.environment.is_empty(), false);
        assert_eq!(game.environment.len(), NUM_OF_BANDITS);
    }

    #[test]
    fn test_running_game_stohastically() {
        let mut game = BernoulliOneGameLearningAgent::new();

        game.run_one_game();

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
    fn test_random_action_within_the_range_returned() {
        let game = BernoulliOneGameLearningAgent::new();
        let turns = 10_000;
        let expected_mean = (0.0 + (NUM_OF_BANDITS as f64) - 1.0) / 2.0;
        let mut actions_taken = vec![];
        for i in 0..turns {
            // Always select random action
            let action = game.random_action_selection_policy();
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
        let mut game = BernoulliOneGameLearningAgent::new();
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
        let mut game = BernoulliOneGameLearningAgent::new();
        let mut probabilities = vec![0.0; game.num_of_bandits as usize];
        for (index, bandit) in game.environment.iter().enumerate() {
            probabilities[index] = bandit._get_actual_probablity();
        }

        game.q_values = probabilities.clone();

        // Calculate actual Maximum probability
        let max_value = probabilities
            .clone()
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, f64::max);

        // Calculate to which index the maximum probability corresponds to
        // This index denotes a action that should be taken the most often.
        let max_indices: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &value)| value == max_value)
            .map(|(index, _)| index)
            .collect();

        // Simulate playing the game and record actions that are returned to be
        // taken on each turn.
        let mut actions_taken = vec![];
        for i in 0..turns {
            let action = game.epsilon_greedy_action_selection_policy();
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