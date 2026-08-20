// Autoresearch Loop — metric-driven iterative improvement.
//
// Each iteration: run a change, measure a metric, compare against baseline.
// Keep the change if it improves; discard if worse or unchanged.

pub mod metric;
pub mod runner;
pub mod report;

pub use metric::{Metric, MetricValue};
pub use runner::{AutoResearchRunner, AutoResearchConfig};
pub use report::ResearchReport;
