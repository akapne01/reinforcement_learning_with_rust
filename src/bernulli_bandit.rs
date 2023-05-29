use rand::distributions::{ Bernoulli, Distribution };

struct BernulliBandit {
    probability: f64,
    distribution: Bernoulli,
}

impl BernulliBandit {
    fn new(probability: f64) -> Self {
        BernulliBandit {
            probability,
            distribution: Bernoulli::new(probability).unwrap(),
        }
    }

    fn pull(&self) -> bool {
        self.distribution.sample(&mut rand::thread_rng())
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
            assert_eq!(result, false);
        }
    }

    #[test]
    fn test_pulling_always_return_true_when_p_set_to_one() {
        let bandit = BernulliBandit::new(1.0);

        for _ in 0..100 {
            let result = bandit.pull();
            assert_eq!(result, true);
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
            // results.insert(trial, result);
            if result {
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
}
