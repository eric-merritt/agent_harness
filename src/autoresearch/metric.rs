// Metric trait and value types for the autoresearch loop.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A single measured value with a timestamp.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricValue {
	pub value: f64,
	pub timestamp: chrono::DateTime<chrono::Utc>,
	pub iteration: u32,
}

/// Trait for any measurable quantity (tokens/sec, latency, accuracy, etc.).
#[async_trait]
pub trait Metric: Send + Sync {
	/// Human-readable name, e.g. "tokens_per_second".
	fn name(&self) -> &str;

	/// Target value to achieve.
	fn target(&self) -> f64;

	/// Measure the current value.
	async fn measure(&self) -> f64;

	/// Compare two values: returns true if `new` is an improvement over `old`.
	fn is_better(&self, new: f64, old: f64) -> bool {
		// Default: higher is better (override for metrics where lower is better)
		new > old
	}

	/// Check if the current value meets the target.
	fn is_target_met(&self, current: f64) -> bool {
		// Within 5% of target
		(current - self.target()).abs() / self.target().max(1e-9) < 0.05
	}
}

/// Built-in metric: higher is better (e.g. throughput).
pub struct HigherIsBetterMetric {
	pub name: String,
	pub target: f64,
	measure_fn: Box<dyn Fn() -> f64 + Send + Sync>,
}

impl HigherIsBetterMetric {
	pub fn new<F>(name: &str, target: f64, f: F) -> Self
	where
		F: Fn() -> f64 + Send + Sync + 'static,
	{
		Self {
			name: name.to_string(),
			target,
			measure_fn: Box::new(f),
		}
	}
}

#[async_trait]
impl Metric for HigherIsBetterMetric {
	fn name(&self) -> &str {
		&self.name
	}

	fn target(&self) -> f64 {
		self.target
	}

	async fn measure(&self) -> f64 {
		(self.measure_fn)()
	}

	fn is_better(&self, new: f64, old: f64) -> bool {
		new > old
	}
}

/// Built-in metric: lower is better (e.g. latency, error rate).
pub struct LowerIsBetterMetric {
	pub name: String,
	pub target: f64,
	measure_fn: Box<dyn Fn() -> f64 + Send + Sync>,
}

impl LowerIsBetterMetric {
	pub fn new<F>(name: &str, target: f64, f: F) -> Self
	where
		F: Fn() -> f64 + Send + Sync + 'static,
	{
		Self {
			name: name.to_string(),
			target,
			measure_fn: Box::new(f),
		}
	}
}

#[async_trait]
impl Metric for LowerIsBetterMetric {
	fn name(&self) -> &str {
		&self.name
	}

	fn target(&self) -> f64 {
		self.target
	}

	async fn measure(&self) -> f64 {
		(self.measure_fn)()
	}

	fn is_better(&self, new: f64, old: f64) -> bool {
		new < old
	}
}
