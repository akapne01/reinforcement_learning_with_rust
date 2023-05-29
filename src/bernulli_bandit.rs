use rand::distributions::{ Bernoulli, Distribution };
use rand::Rng;

fn generate_uniform_random_number() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen()
}

pub fn generate_random_number_in_range(min: i32, max: i32) -> i32 {
    if min > max {
        panic!("Minimum number cannot be bigger than maximum number! Please check your values!");
    }
    rand::thread_rng().gen_range(min..=max)
}

#[derive(PartialEq, Debug)]
pub struct BernulliBandit {
    probability: f64,
    distribution: Bernoulli,
}

impl BernulliBandit {
    const IS_VERBOSE_MODE: bool = true;

    pub fn new(probability: f64) -> Self {
        if Self::IS_VERBOSE_MODE {
            println!("# Creating Bernulli Bandit with probability: {probability} #");
        }
        BernulliBandit {
            probability,
            distribution: Bernoulli::new(probability).unwrap(),
        }
    }

    pub fn new_random() -> Self {
        let probability = generate_uniform_random_number();
        BernulliBandit::new(probability)
    }

    pub fn new_as_vector(number_of_bandits: usize) -> Vec<BernulliBandit> {
        let mut bandits = Vec::new();
        for index in 0..number_of_bandits {
            bandits.insert(index, BernulliBandit::new_random());
        }
        bandits
    }

    pub fn pull(&self) -> f64 {
        let result = self.distribution.sample(&mut rand::thread_rng());
        match result {
            true => { 1.0 }
            false => { 0.0 }
        }
    }

    pub fn get_probablity(&self) -> f64 {
        self.probability
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_that_random_number_between_zero_and_one() {
        for _ in 0..1000 {
            let number = generate_uniform_random_number();
            assert!(number >= 0.0);
            assert!(number <= 1.0);
        }
    }

    #[test]
    #[should_panic(
        expected = "Minimum number cannot be bigger than maximum number! Please check your values!"
    )]
    fn test_generate_random_number_when_max_smaller_than_min() {
        generate_random_number_in_range(100, 1);
    }

    #[test]
    fn test_generate_random_number_in_range() {
        let min = 1;
        let max = 10;
        for _ in 0..1000 {
            let number = generate_random_number_in_range(min, max);
            assert!(number >= min);
            assert!(number <= max);
        }
    }

    #[test]
    fn test_new_bandit_creation() {
        let k_bandit = BernulliBandit::new(0.5);

        assert_eq!(k_bandit.probability, 0.5);
    }

    #[test]
    fn test_pulling_always_returns_false_when_p_set_zero() {
        let bandit = BernulliBandit::new(0.0);

        for _ in 0..100 {
            let result = bandit.pull();
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn test_pulling_always_return_true_when_p_set_to_one() {
        let bandit = BernulliBandit::new(1.0);

        for _ in 0..100 {
            let result = bandit.pull();
            assert_eq!(result, 1.0);
        }
    }

    #[test]
    fn test_pulling_when_p_is_one_half() {
        let mut success_counter = 0;
        let mut failure_counter = 0;
        let bandit = BernulliBandit::new(0.5);
        let expected_range = 450..550; // 1000 trials

        for _ in 0..1000 {
            let result = bandit.pull();
            if result == 1.0 {
                success_counter += 1;
            } else {
                failure_counter += 1;
            }
        }

        assert!(
            expected_range.contains(&success_counter),
            "When 1000 trials and p=0.5, then true values expected to be in range 450..550"
        );
        assert!(
            expected_range.contains(&failure_counter),
            "When 1000 trials and p=0.5, then false values expected to be in range 450..550"
        );
    }

    #[test]
    fn test_new_vector() {
        let vector = BernulliBandit::new_as_vector(10);

        assert_eq!(vector.is_empty(), false);
        assert_eq!(vector.len(), 10);
        assert_ne!(
            vector[0].probability,
            vector[1].probability,
            "Bernulli Bandits each have a different probability"
        );
    }
}
