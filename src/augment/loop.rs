// ── Loop augment ───────────────────────────────────────────────────────────────

/// Trigger condition for a loop augment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LoopTrigger {
	/// Fire every N seconds
	Interval { secs: u64 },
	/// Fire when a condition prompt evaluates true
	Conditional { prompt: String },
	/// Fire on a specific event
	EventDriven { event: String },
	/// Self-paced — the loop decides when to re-fire
	SelfPaced,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopAugment {
	pub id: Uuid,
	pub name: String,
	pub prompt: String,
	pub trigger: LoopTrigger,
	pub max_iterations: Option<u32>,
	pub is_active: bool,
}