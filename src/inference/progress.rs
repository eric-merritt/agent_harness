// Shared progress tracker — used by model loading code and UI.
// No UI dependencies, so it can be imported from anywhere.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, RwLock};

/// Thread-safe progress tracker — updated by the loading thread, read by the render loop.
#[derive(Clone)]
pub struct LoadingProgress {
	pub percentage: Arc<AtomicU8>,
	pub status: Arc<RwLock<String>>,
	pub done: Arc<AtomicU8>, // 0=loading, 1=done, 2=failed
}

impl LoadingProgress {
	pub fn new() -> Self {
		Self {
			percentage: Arc::new(AtomicU8::new(0)),
			status: Arc::new(RwLock::new("Starting...".to_string())),
			done: Arc::new(AtomicU8::new(0)),
		}
	}

	pub fn set(&self, pct: u8, msg: &str) {
		self.percentage.store(pct.min(100), Ordering::SeqCst);
		*self.status.write().unwrap_or_else(|p| p.into_inner()) = msg.to_string();
	}

	pub fn finish(&self) {
		self.percentage.store(100, Ordering::SeqCst);
		*self.status.write().unwrap_or_else(|p| p.into_inner()) = "Done".to_string();
		self.done.store(1, Ordering::SeqCst);
	}

	pub fn fail(&self, msg: &str) {
		*self.status.write().unwrap_or_else(|p| p.into_inner()) = msg.to_string();
		self.done.store(2, Ordering::SeqCst);
	}

	pub fn is_done(&self) -> bool {
		self.done.load(Ordering::SeqCst) != 0
	}

	pub fn is_failed(&self) -> bool {
		self.done.load(Ordering::SeqCst) == 2
	}

	pub fn get_pct(&self) -> u8 {
		self.percentage.load(Ordering::SeqCst)
	}

	pub fn get_status(&self) -> String {
		self.status
			.read()
			.unwrap_or_else(|p| p.into_inner())
			.clone()
	}
}

impl Default for LoadingProgress {
	fn default() -> Self {
		Self::new()
	}
}
