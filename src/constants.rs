/// If true, prints in console additional inforamtion.
pub const IS_VERBOSE_MODE: bool = false;
/// If true, prints out the action taken and reward receieved for each turn in the game.
pub const PRINT_EACH_STEP: bool = false;
/// Represent number of independent games to play.
pub const NUM_OF_GAMES_TO_PLAY: usize = 100;
/// Represents the number of slot machines being played in k-armed bandit problem, it is the number k.
pub const NUM_OF_BANDITS: usize = 10;
/// Represents the number of turns in one game.
pub const NUM_OF_TURNS_IN_A_GAME: usize = 100_000;
/// Represents the probability with which random action is selected. It reflects the probability
/// with which agent explores the action space. The probability that agent takes the action that
/// exploits the knowledge that it has learned is (1-EPSILON).
/// Epsilon is expected to be in bounds 0 <= EPSILON <= 1. When EPSILON = 0, agent always takes
/// greedy action and explits the knowledge that it has. When EPSILON = 1, agent always takes
/// exploratory action and learns more about action space based on rewards recieved.
pub const EPSILON: f64 = 1.0;
/// Is a stepsize parameter. This parameter really comes from mean average method where it represents
/// 1/n where n is the number of turns taken in the game. It allows to adjust how much weight is given
/// to each step.
pub const ALPHA: f64 = 0.0099;
pub const USE_AVERAGE_VALUE_UPDATE: bool = true;
pub const SHOULD_DECKAY_EPSILON: bool = true;
