// Research report — summarizes all iterations.

use serde::{Deserialize, Serialize};

/// Result of a single iteration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationResult {
	pub iteration: u32,
	pub value: f64,
	pub improved: bool,
	pub kept: bool,
	pub notes: String,
}

/// Full report of an autoresearch run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResearchReport {
	pub baseline: f64,
	pub iterations: Vec<IterationResult>,
	pub total_improvements: usize,
	pub terminated: String,
}

impl ResearchReport {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn finalize(&mut self) {
		// Compute summary stats
		let values: Vec<f64> = self.iterations.iter().map(|i| i.value).collect();
		if !values.is_empty() {
			let best = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
			let worst = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
			self.terminated = format!(
				"{} (best={:.4}, worst={:.4}, improvements={})",
				self.terminated, best, worst, self.total_improvements
			);
			log::debug!(
				"Autoresearch report finalized — best={:.4}, worst={:.4}, improvements={}",
				best,
				worst,
				self.total_improvements
			);
		}
	}

	/// Return a human-readable summary.
	pub fn summary(&self) -> String {
		format!(
			"Autoresearch Report:\n\
             Baseline: {:.4}\n\
             Iterations: {}\n\
             Improvements kept: {}\n\
             Terminated: {}",
			self.baseline,
			self.iterations.len(),
			self.total_improvements,
			self.terminated,
		)
	}
}
