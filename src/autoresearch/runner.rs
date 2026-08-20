// Autoresearch runner — drives iterations, measures metrics, keeps or discards changes.

use std::sync::Arc;
use tokio::time::{Duration, timeout};

use super::metric::{Metric, MetricValue};
use super::report::{ResearchReport, IterationResult};

/// Configuration for an autoresearch run.
#[derive(Clone, Debug)]
pub struct AutoResearchConfig {
    pub max_iterations: u32,
    /// Seconds per iteration (time-box).
    pub iteration_timeout_secs: u64,
    /// Minimum improvement percentage to keep a change (default 1%).
    pub min_improvement_pct: f64,
}

impl Default for AutoResearchConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            iteration_timeout_secs: 600, // 10 min
            min_improvement_pct: 1.0,
        }
    }
}

/// Runs the autoresearch loop.
pub struct AutoResearchRunner<M: Metric> {
    metric: Arc<M>,
    config: AutoResearchConfig,
    baseline: Option<f64>,
    history: Vec<MetricValue>,
    report: ResearchReport,
    running: bool,
}

impl<M: Metric> AutoResearchRunner<M> {
    pub fn new(metric: M, config: AutoResearchConfig) -> Self {
        Self {
            metric: Arc::new(metric),
            config,
            baseline: None,
            history: Vec::new(),
            report: ResearchReport::new(),
            running: false,
        }
    }

    /// Run the full loop.
    pub async fn run(&mut self) -> ResearchReport {
        log::info!(
            "Autoresearch: starting run — metric='{}', max_iterations={}",
            self.metric.name(),
            self.config.max_iterations
        );
        self.running = true;

        // Establish baseline
        let baseline = self.measure_iteration(0).await;
        self.baseline = Some(baseline);
        self.history.push(self.make_value(0, baseline));
        self.report.baseline = baseline;
        log::info!("Autoresearch: baseline established — {}={:.4}", self.metric.name(), baseline);

        let iter_timeout = Duration::from_secs(self.config.iteration_timeout_secs);

        for i in 1..=self.config.max_iterations {
            if !self.running {
                log::info!("Autoresearch: manual stop at iteration {}", i);
                self.report.terminated = "manual_stop".to_string();
                break;
            }

            log::info!("Autoresearch: iteration {} starting", i);

            // Time-box the iteration
            let iter_result = timeout(iter_timeout, self.run_iteration(i)).await;

            match iter_result {
                Ok(result) => {
                    log::debug!(
                        "Autoresearch: iteration {} — value={:.4}, improved={}, kept={}",
                        i, result.value, result.improved, result.kept
                    );
                    self.history.push(self.make_value(i, result.value));
                    self.report.iterations.push(IterationResult {
                        iteration: i,
                        value: result.value,
                        improved: result.improved,
                        kept: result.kept,
                        notes: result.notes.clone(),
                    });

                    if !result.improved {
                        log::warn!(
                            "Autoresearch: iteration {} — metric regression ({}={:.4} vs baseline={:.4})",
                            i, self.metric.name(), result.value, baseline
                        );
                    }

                    if result.kept {
                        self.report.total_improvements += 1;
                        log::info!(
                            "Autoresearch: iteration {} — change kept ({}={:.4})",
                            i, self.metric.name(), result.value
                        );
                    }

                    // Stop if target is met
                    if self.metric.is_target_met(result.value) {
                        log::info!(
                            "Autoresearch: target met at iteration {} ({}={:.4}, target={:.4})",
                            i, self.metric.name(), result.value, self.metric.target()
                        );
                        self.report.terminated = "target_met".to_string();
                        break;
                    }
                }
                Err(_) => {
                    log::warn!(
                        "Autoresearch: iteration {} timed out after {}s",
                        i, self.config.iteration_timeout_secs
                    );
                    self.report.iterations.push(IterationResult {
                        iteration: i,
                        value: baseline,
                        improved: false,
                        kept: false,
                        notes: "iteration_timeout".to_string(),
                    });
                }
            }
        }

        if self.report.terminated.is_empty() {
            log::info!(
                "Autoresearch: reached max iterations ({}) — terminating",
                self.config.max_iterations
            );
            self.report.terminated = "max_iterations_reached".to_string();
        }

        self.running = false;
        self.report.finalize();
        log::info!(
            "Autoresearch: run complete — improvements={}, terminated='{}'",
            self.report.total_improvements, self.report.terminated
        );
        self.report.clone()
    }

    /// Measure once and return the value.
    async fn measure_iteration(&self, _iteration: u32) -> f64 {
        self.metric.as_ref().measure().await
    }

    /// Run a single iteration: apply change, measure, decide.
    async fn run_iteration(&self, iteration: u32) -> IterationDecision {
        // TODO: In a full implementation, this would:
        // 1. Create a feature branch
        // 2. Apply a change (guided by LLM)
        // 3. Measure the metric
        // 4. Compare to baseline
        // 5. Keep or discard the change

        let current = self.metric.as_ref().measure().await;
        let baseline = self.baseline.unwrap_or(current);

        let improved = self.metric.as_ref().is_better(current, baseline);
        let improvement_pct = if baseline.abs() > 1e-9 {
            ((current - baseline).abs() / baseline.abs()) * 100.0
        } else {
            0.0
        };

        log::debug!(
            "Autoresearch: metric comparison — {}={:.4} vs baseline={:.4} (improved={}, improvement_pct={:.2}%)",
            self.metric.name(), current, baseline, improved, improvement_pct
        );

        let meets_threshold = improvement_pct >= self.config.min_improvement_pct;
        let kept = improved && meets_threshold;

        IterationDecision {
            iteration,
            value: current,
            improved,
            kept,
            notes: if kept {
                format!("improved by {:.2}%", improvement_pct)
            } else if improved {
                format!("improved but below threshold ({:.2}% < {:.2}%)", improvement_pct, self.config.min_improvement_pct)
            } else {
                "no improvement".to_string()
            },
        }
    }

    fn make_value(&self, iteration: u32, value: f64) -> MetricValue {
        MetricValue {
            value,
            timestamp: chrono::Utc::now(),
            iteration,
        }
    }

    /// Stop the loop early.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Get the metric history.
    pub fn history(&self) -> &[MetricValue] {
        &self.history
    }
}

#[derive(Clone, Debug)]
struct IterationDecision {
    iteration: u32,
    value: f64,
    improved: bool,
    kept: bool,
    notes: String,
}
