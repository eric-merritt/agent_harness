// Autoresearch Loop — metric-driven iterative improvement.
//
// Each iteration: run a change, measure a metric, compare against baseline.
// Keep the change if it improves; discard if worse or unchanged.

pub mod metric;
pub mod report;
pub mod runner;

pub use metric::{Metric, MetricValue};
pub use report::ResearchReport;
pub use runner::{AutoResearchConfig, AutoResearchRunner};
