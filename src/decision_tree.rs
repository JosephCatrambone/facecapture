
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DecisionTree {
	feature: usize,
	threshold: f32,
	class: bool,
	class_probability: f32,
	
	left_subtree: Option<Box::<DecisionTree>>,
	right_subtree: Option<Box::<DecisionTree>>,
}

impl DecisionTree {
	pub fn new() -> Self {
		DecisionTree { ..Default::default() }
	}
	
	pub fn predict(&self, example: &Vec<f32>) -> bool {
		let (cls, _) = self.predict_with_confidence(example);
		cls
	}
	
	pub fn predict_with_confidence(&self, example: &Vec<f32>) -> (bool, f32) {
		if let (Some(ls), Some(rs)) = (&self.left_subtree, &self.right_subtree) {
			if example[self.feature] < self.threshold {
				ls.predict_with_confidence(example)
			} else {
				rs.predict_with_confidence(example)
			}
		} else {
			(self.class, self.class_probability)
		}
	}
	
	pub fn train(&mut self, examples: &Vec<&Vec<f32>>, labels:&Vec<bool>, max_depth:u32) {
		self.threshold = 0.5f32;
		
		// If this tree is 100% uniform (perfectly decides the data), we don't have to do more work.
		// Check if this is the case.
		let mut num_positive = 0;
		let mut num_negative = 0;
		for c in labels {
			if *c {
				num_positive += 1;
			} else {
				num_negative += 1;
			}
		}
		
		// Find class purity.
		if num_positive > num_negative {
			self.class = true;
			self.class_probability = num_positive as f32 / (num_positive+num_negative) as f32;
		} else {
			self.class = false;
			self.class_probability = num_negative as f32 / (num_positive+num_negative) as f32;
		}
		
		// We might not be able to go further.
		if num_positive == 0 || num_negative == 0 || max_depth == 0 {
			return;
		}
		
		// We actually do have work to do.  Find the feature which maximizes information gain.
		let mut best_gini_impurity = 1000f32;
		let mut best_feature = 0;
		for candidate_feature in 0..examples[0].len() {
			// TODO: Calculate midpoint.
			let impurity = gini_impurity(examples, labels, candidate_feature, self.threshold);
			if impurity < best_gini_impurity {
				best_gini_impurity = impurity;
				best_feature = candidate_feature;
			}
		}
		// Now we have our optimal split.
		self.feature = best_feature;
		
		// Make the left examples.
		let mut left_examples = Vec::<&Vec<f32>>::new();
		let mut right_examples = Vec::<&Vec<f32>>::new();
		let mut left_labels = Vec::<bool>::new();
		let mut right_labels = Vec::<bool>::new();
		for (x, y) in examples.into_iter().zip(labels) {
			if x[self.feature] < self.threshold {
				left_examples.push(x);
				left_labels.push(*y);
			} else {
				right_examples.push(x);
				right_labels.push(*y);
			}
		}
		
		let mut left_sub = Box::new(DecisionTree::new());
		left_sub.train(&left_examples, &left_labels, max_depth-1);
		self.left_subtree = Some(left_sub);
		
		let mut right_sub = Box::new(DecisionTree::new());
		right_sub.train(&right_examples, &right_labels, max_depth-1);
		self.right_subtree = Some(right_sub);
	}
}

fn gini_impurity(examples: &Vec<&Vec<f32>>, labels: &Vec<bool>, feature:usize, thresh:f32) -> f32 {
	let mut count_true_if_feature = 0;
	let mut count_false_if_feature = 0;
	let mut count_true_ifnot_feature = 0;
	let mut count_false_ifnot_feature = 0;
	
	for (x,y) in examples.iter().zip(labels) {
		// Gini impurity = 1.0 - (# T / total)^2 - (# F / total)^2
		if x[feature] < thresh {
			// ifnot_feature
			if *y {
				count_true_ifnot_feature += 1;
			} else {
				count_false_ifnot_feature += 1;
			}
		} else {
			// if_feature
			if *y {
				count_true_if_feature += 1;
			} else {
				count_false_if_feature += 1;
			}
		}
	}
	
	let p_true_given_feature = count_true_if_feature as f32 / (1e-6 + (count_true_if_feature + count_false_if_feature) as f32);
	let p_false_given_feature = count_false_if_feature as f32 / (1e-6 + (count_true_if_feature + count_false_if_feature) as f32);
	let p_true_given_not_feature = count_true_ifnot_feature as f32 / (1e-6 + (count_true_ifnot_feature + count_false_ifnot_feature) as f32);
	let p_false_given_not_feature = count_false_ifnot_feature as f32 / (1e-6 + (count_true_ifnot_feature + count_false_ifnot_feature) as f32);
	
	let gini_impurity_with_feature = 1.0 - (p_true_given_feature*p_true_given_feature) - (p_false_given_feature*p_false_given_feature);
	let gini_impurity_without_feature = 1.0 - (p_true_given_not_feature*p_true_given_not_feature) - (p_false_given_not_feature*p_false_given_not_feature);
	
	let total_entries = (count_true_if_feature + count_false_if_feature + count_true_ifnot_feature + count_false_ifnot_feature) as f32;
	
	gini_impurity_with_feature*((count_true_if_feature+count_false_if_feature) as f32/total_entries) + gini_impurity_without_feature*((count_true_ifnot_feature+count_false_ifnot_feature) as f32/total_entries)
}

#[cfg(test)]
mod tests {
	use super::*;
	
	macro_rules! assert_approx_eq {
		($a:expr, $b:expr) => {
			assert!(($a as f32 - $b as f32).abs() < 1e-3f32, "{} !~= {}", $a, $b);
		};
	}
	
	#[test]
	fn test_tree_sanity() {
		let mut dt = DecisionTree::new();
		// Is this game for children?
		// Feat: exploding heads, talking animals, chainsaws, shotgun blasts
		let doom = vec![1f32, 0f32, 1f32, 1f32];
		let animal_crossing = vec![0f32, 1f32, 1f32, 0f32];
		let gta = vec![0f32, 0f32, 1f32, 1f32];
		let pokemon = vec![0f32, 1f32, 0f32, 0f32];
		let quake = vec![1f32, 0f32, 0f32, 1f32];
		let fez = vec![0f32, 0f32, 0f32, 0f32];
		let examples = vec![&doom, &animal_crossing, &gta, &pokemon, &quake, &fez];
		let labels = vec![false, true, false, true, false, true];
		dt.train(&examples, &labels, 4);
		let wolfenstein = vec![1f32, 0f32, 1f32, 1f32];
		let terraria = vec![0f32, 1f32, 0f32, 0f32];
		assert!(!dt.predict(&wolfenstein));
		assert!(dt.predict(&terraria));
	}
}