use rand::distributions::{ Bernoulli, Distribution };
use rand::Rng;

use crate::constants::IS_VERBOSE_MODE;

/// Generates random number in range: [0; 1]
pub fn generate_uniform_random_number() -> f64 {
    rand::thread_rng().gen()
}

/// Generates random number in range: [min, max)
pub fn generate_random_number_in_range(min: usize, max: usize) -> usize {
    assert!(min < max, "Minimum number cannot be bigger than maximum number!");
    rand::thread_rng().gen_range(min..max)
}

/// BernulliBandit object represents a leaver in the slot machine.
/// It has a certain proability of winning or loosing that is constant
/// and does not change over time.
/// This probability is not known beforehand.
#[derive(PartialEq, Debug, Clone)]
pub struct BernoulliBanditEnvironment {
    probability: f64,
    distribution: Bernoulli,
}

impl BernoulliBanditEnvironment {
    /// This method allows to create a new Bernulli Bandit object with custom
    /// proabability.
    pub fn new(probability: f64) -> Self {
        assert!((0.0..=1.0).contains(&probability), "Probability must be in the range [0, 1].");
        if IS_VERBOSE_MODE {
            println!("# Creating Bernulli Bandit with probability: {probability} #");
        }
        BernoulliBanditEnvironment {
            probability,
            distribution: Bernoulli::new(probability).unwrap(),
        }
    }

    /// This method allows to create a new Bernulli Bandit object with
    /// random probability.
    pub fn new_random() -> Self {
        let probability = generate_uniform_random_number();
        BernoulliBanditEnvironment::new(probability)
    }

    /// Generates a specified number of new random proability BernoulliBandit objects
    /// and returns them as vector
    pub fn new_as_vector(number_of_bandits: usize) -> Vec<BernoulliBanditEnvironment> {
        (0..number_of_bandits).map(|_| BernoulliBanditEnvironment::new_random()).collect()
    }

    /// Each pull represents pulling a leaver similar like in the slot machine.
    /// Pulling returns reward that is of Bernoulli distribution, meaning that
    /// 1.0 represent win, and 0.0 represent loss.
    pub fn step(&self) -> f64 {
        let result = self.distribution.sample(&mut rand::thread_rng());
        match result {
            true => 1.0,
            false => 0.0,
        }
    }

    /// Added ONLY for purposes of collecting statistics about the game.
    /// Since proability is set randomly, we need a way to obtain how the
    /// actual probability was so it can be compared to what the agent
    /// has learned.
    pub fn _get_actual_probablity(&self) -> f64 {
        self.probability
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_generate_uniform_random_number_within_range() {
        for _ in 0..1000 {
            let number = generate_uniform_random_number();

            assert!(
                (0.0..=1.0).contains(&number),
                "Generated random number is not within the range [0, 1]: {}",
                number
            );
        }
    }

    #[test]
    fn test_generate_random_number_in_range_min_less_than_max() {
        let min = 1;
        let max = 10;
        for _ in 0..1000 {
            let number = generate_random_number_in_range(min, max);
            assert!(
                (min..max).contains(&number),
                "Generated random number is not within the range [{}, {}): {}",
                min,
                max,
                number
            );
        }
    }

    #[test]
    #[should_panic(expected = "Minimum number cannot be bigger than maximum number!")]
    fn test_generate_random_number_when_max_smaller_than_min_max_not_inclusive() {
        generate_random_number_in_range(100, 1);
    }

    #[test]
    fn test_create_bernoulli_bandit_with_valid_probability() {
        let bandit = BernoulliBanditEnvironment::new(0.5);
        assert_eq!(bandit.probability, 0.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be in the range [0, 1].")]
    fn test_create_bernoulli_bandit_with_probability_greater_than_one() {
        BernoulliBanditEnvironment::new(1.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be in the range [0, 1].")]
    fn test_create_bernoulli_bandit_with_probability_less_than_zero() {
        BernoulliBanditEnvironment::new(-0.5);
    }

    #[test]
    fn test_bernoulli_bandit_pull_always_returns_zero_when_probability_is_zero() {
        let bandit = BernoulliBanditEnvironment::new(0.0);
        for _ in 0..100 {
            let result = bandit.step();
            assert_eq!(result, 0.0, "Pull result is not zero when probability is zero");
        }
    }

    #[test]
    fn test_bernoulli_bandit_pull_always_returns_one_when_probability_is_one() {
        let bandit = BernoulliBanditEnvironment::new(1.0);
        for _ in 0..100 {
            let result = bandit.step();
            assert_eq!(result, 1.0, "Pull result is not one when probability is one");
        }
    }

    #[test]
    fn test_generate_random_number_in_range_max_not_inclusive() {
        let min = 1;
        let max = 10;
        for _ in 0..1000 {
            let number = generate_random_number_in_range(min, max);
            assert!(number >= min);
            assert!(number < max);
        }
    }

    #[test]
    fn test_bernoulli_bandit_pull_when_probability_is_half() {
        let mut success_counter = 0;
        let mut failure_counter = 0;
        let bandit = BernoulliBanditEnvironment::new(0.5);
        let expected_range = 450..550; // 1000 trials

        for _ in 0..1000 {
            let result = bandit.step();
            if result == 1.0 {
                success_counter += 1;
            } else {
                failure_counter += 1;
            }
        }

        assert!(
            expected_range.contains(&success_counter),
            "Number of successes not within the expected range [450, 550): {}",
            success_counter
        );

        assert!(
            expected_range.contains(&failure_counter),
            "Number of failures not within the expected range [450, 550): {}",
            failure_counter
        );
    }

    #[test]
    fn test_create_random_bernoulli_bandit() {
        let bandit = BernoulliBanditEnvironment::new_random();
        assert!(
            (0.0..=1.0).contains(&bandit.probability),
            "Randomly created bandit has an invalid probability: {}",
            bandit.probability
        );
    }

    #[test]
    fn test_create_vector_of_bernoulli_bandits() {
        let vector = BernoulliBanditEnvironment::new_as_vector(10);

        assert_eq!(vector.is_empty(), false);
        assert_eq!(vector.len(), 10);

        let mut probabilities: Vec<f64> = vector
            .iter()
            .map(|bandit| bandit.probability)
            .collect();
        probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        probabilities.dedup();
        let unique_probabilities = probabilities.clone();

        assert_eq!(
            unique_probabilities.len(),
            10,
            "Bernoulli Bandits do not have unique probabilities"
        );
    }
}
