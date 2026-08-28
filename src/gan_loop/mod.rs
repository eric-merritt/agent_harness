// Generative Adversarial Network (GAN) Loop — three-agent orchestration:
//   Planner → Generator → Evaluator → (repeat until all tasks complete)
//
// This module provides the state machine and coordination logic.

pub mod evaluator;
pub mod generator;
pub mod planner;
pub mod state;

pub use state::{GANLoopConfig, GANLoopState, GANTerminationReason, TaskItem, TaskStatus};

/// Run one full GAN loop: plan → generate → evaluate → repeat.
///
/// Returns when all tasks are complete, max iterations is reached, or timeout fires.
pub async fn run_loop(mut loop_state: GANLoopState, config: GANLoopConfig) -> GANLoopState {
	use tokio::time::{Duration, timeout};

	log::info!(
		"GAN loop starting — goal: '{}', max_iterations: {}",
		loop_state.goal,
		config.max_iterations
	);
	loop_state.start();

	let loop_timeout = Duration::from_secs(config.timeout_secs);
	let _ = timeout(loop_timeout, async {
		// Phase 1: Plan
		log::info!("GAN loop phase 1: planning");
		planner::plan(&mut loop_state).await;
		log::info!(
			"GAN loop planning complete — {} tasks",
			loop_state.tasks.len()
		);

		// Phase 2+: Generate → Evaluate cycles
		while loop_state.running && !loop_state.is_finished() {
			loop_state.current_iteration += 1;
			log::info!(
				"GAN loop iteration {} — {}% complete",
				loop_state.current_iteration,
				loop_state.completion_percentage()
			);

			// Check max iterations
			if loop_state.current_iteration > config.max_iterations {
				log::warn!(
					"GAN loop hit max iterations ({}) — terminating",
					loop_state.current_iteration
				);
				loop_state.terminate(GANTerminationReason::MaxIterationsReached(
					loop_state.current_iteration,
				));
				break;
			}

			// Generate
			generator::generate(&mut loop_state).await;

			// Evaluate
			evaluator::evaluate(&mut loop_state).await;

			let counts = loop_state.status_counts();
			log::debug!(
				"GAN loop iteration {} done — status: {:?}",
				loop_state.current_iteration,
				counts
			);

			// Check if finished after this round
			if loop_state.is_finished() {
				loop_state.terminate(GANTerminationReason::AllComplete);
				break;
			}
		}
	})
	.await;

	// If we exited because of the outer timeout
	if loop_state.running {
		log::warn!("GAN loop timed out after {}s", config.timeout_secs);
		loop_state.terminate(GANTerminationReason::Timeout);
	}

	log::info!(
		"GAN loop finished — reason: {:?}, iterations: {}, completion: {:.1}%",
		loop_state.terminated,
		loop_state.current_iteration,
		loop_state.completion_percentage()
	);
	loop_state
}
