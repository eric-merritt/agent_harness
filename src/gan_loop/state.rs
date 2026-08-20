// GAN Loop state: task list, completion tracking, evaluator notes, config

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Per-item status in the GAN loop.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    /// Not yet evaluated — the generator has proposed work but evaluator hasn't checked.
    Pending,
    /// Generator claims done; awaiting evaluation.
    AwaitingReview,
    /// Evaluator confirmed complete.
    Complete,
    /// Evaluator rejected — notes explain what's wrong; generator must retry.
    Rejected { notes: String },
}

/// A single task in the loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskItem {
    pub id: Uuid,
    pub description: String,
    pub status: TaskStatus,
    /// What the generator produced this iteration (code, text, output).
    pub generation: Option<String>,
    /// Evaluator's notes — empty unless Rejected.
    pub evaluator_notes: Option<String>,
    /// How many iterations this task has gone through.
    pub iteration_count: u32,
}

impl TaskItem {
    pub fn new(description: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            status: TaskStatus::Pending,
            generation: None,
            evaluator_notes: None,
            iteration_count: 0,
        }
    }

    /// Returns true if the task is definitively done.
    pub fn is_resolved(&self) -> bool {
        matches!(self.status, TaskStatus::Complete)
    }

    /// Returns true if the task is actively blocked on the generator.
    pub fn needs_generation(&self) -> bool {
        matches!(self.status, TaskStatus::Pending | TaskStatus::Rejected { .. })
    }

    /// Returns true if the task is blocked on the evaluator.
    pub fn needs_evaluation(&self) -> bool {
        matches!(self.status, TaskStatus::AwaitingReview)
    }
}

/// Configuration for the GAN loop run.
#[derive(Clone, Debug)]
pub struct GANLoopConfig {
    /// Maximum number of plan→generate→evaluate cycles.
    pub max_iterations: u32,
    /// Timeout in seconds for the entire loop.
    pub timeout_secs: u64,
    /// Timeout in seconds per individual generation step.
    pub generation_timeout_secs: u64,
    /// Require ≥N evaluators to agree (for future multi-evaluator support).
    pub agreement_threshold: usize,
}

impl Default for GANLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            timeout_secs: 3600, // 1 hour
            generation_timeout_secs: 300, // 5 min per generation
            agreement_threshold: 1,
        }
    }
}

/// Full mutable state of a GAN loop.
#[derive(Clone, Debug, Default)]
pub struct GANLoopState {
    /// The original user goal or question.
    pub goal: String,
    /// Tasks decomposed by the planner.
    pub tasks: Vec<TaskItem>,
    /// Current iteration number (0 = planning, 1+ = generate/evaluate cycles).
    pub current_iteration: u32,
    /// Overall loop status.
    pub running: bool,
    /// Termination reason (set when loop ends).
    pub terminated: Option<GANTerminationReason>,
}

#[derive(Clone, Debug)]
pub enum GANTerminationReason {
    AllComplete,
    MaxIterationsReached(u32),
    Timeout,
    ManualStop,
}

impl GANLoopState {
    pub fn new(goal: String) -> Self {
        Self {
            goal,
            tasks: Vec::new(),
            current_iteration: 0,
            running: false,
            terminated: None,
        }
    }

    /// Percentage of tasks that are Complete (0.0–100.0).
    pub fn completion_percentage(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        let done = self.tasks.iter().filter(|t| t.is_resolved()).count() as f64;
        (done / self.tasks.len() as f64) * 100.0
    }

    /// Returns true if the loop has nothing left to do.
    pub fn is_finished(&self) -> bool {
        self.tasks.iter().all(|t| t.is_resolved())
    }

    /// Count by status.
    pub fn status_counts(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for task in &self.tasks {
            let key = match task.status {
                TaskStatus::Pending => "pending",
                TaskStatus::AwaitingReview => "awaiting_review",
                TaskStatus::Complete => "complete",
                TaskStatus::Rejected { .. } => "rejected",
            };
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Mark the loop as started.
    pub fn start(&mut self) {
        self.running = true;
        log::debug!("GAN state: loop started");
    }

    /// Terminate the loop with a reason.
    pub fn terminate(&mut self, reason: GANTerminationReason) {
        log::debug!("GAN state: terminating — reason: {:?}", reason);
        self.running = false;
        self.terminated = Some(reason);
    }
}
