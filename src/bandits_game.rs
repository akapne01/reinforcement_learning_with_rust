use crate::bernulli_bandit::{ BernulliBandit, generate_random_number_in_range };

#[derive(PartialEq, Debug)]
pub struct BanditsGame {
    no_of_bandits: i32,
    no_of_trials: i32,
    bandits: Vec<BernulliBandit>,
    results: Option<Vec<(usize, f64)>>,
}

impl BanditsGame {
    const IS_VERBOSE_MODE: bool = true;

    /// Initalize the game allowing to specify the number of multi-arm bandits
    /// and number of trials that you would like run.
    pub fn new(no_of_bandits: i32, no_of_trials: i32) -> Self {
        BanditsGame {
            no_of_bandits,
            no_of_trials,
            bandits: BernulliBandit::new_as_vector(no_of_bandits as usize),
            results: None,
        }
    }

    pub fn run_stochastic(&mut self) {
        let mut results: Vec<(usize, f64)> = vec![(0, 0.0); self.no_of_trials as usize];

        for trial in 0..self.no_of_trials as usize {
            let bandit_number = generate_random_number_in_range(0, self.no_of_bandits - 1) as usize;
            results[trial] = (bandit_number, self.bandits[bandit_number].pull());
            if Self::IS_VERBOSE_MODE {
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

    pub fn get_actual_probabilities(&self) -> Vec<f64> {
        let mut probabilities = vec![0.0; self.no_of_bandits as usize];
        for (index, bandit) in self.bandits.iter().enumerate() {
            probabilities[index] = bandit.get_probablity();
        }
        probabilities
    }

    pub fn print_statistics(&self) {
        if self.results.is_none() {
            println!("Results not available! Generate results first!!!");
            return;
        }
        let mut bandits_frequency = vec![0; self.no_of_bandits as usize];
        let mut bandits_rewards = vec![0.0; self.no_of_bandits as usize];
        for &(bandit, reward) in self.results.as_ref().unwrap().into_iter() {
            bandits_frequency[bandit] += 1;
            bandits_rewards[bandit] += reward;
        }
        let probabilities = self.get_actual_probabilities();
        println!("### Printing Bandit statistics! ###");
        for i in 0..self.no_of_bandits as usize {
            let frequency = bandits_frequency[i];
            let total_reward = bandits_rewards[i];
            let probability = probabilities[i];
            let mean_reward = total_reward / (frequency as f64);
            println!(
                "Bandit: {} \t Freq: {} \t Total Rwrd: {} \t Mean Rwrd: {} \t p: {}",
                i,
                frequency,
                total_reward,
                mean_reward,
                probability
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_creation_of_game() {
        let game = BanditsGame::new(10, 1000);

        assert_eq!(game.no_of_bandits, 10);
        assert_eq!(game.no_of_trials, 1000);
        assert_eq!(game.bandits.is_empty(), false);
        assert_eq!(game.bandits.len(), 10);
    }
}
