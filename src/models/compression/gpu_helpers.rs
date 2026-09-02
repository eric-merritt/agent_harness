/// Work-pool entry: bits 0–29 = block index, bit 30 = done, bit 31 = claimed.
#[derive(Clone, Copy, Default)]
pub struct WorkPoolEntry(pub u32);

impl WorkPoolEntry {
	pub const DONE_BIT: u32 = 1 << 30;
	pub const CLAIM_BIT: u32 = 1 << 31;
	pub const INDEX_MASK: u32 = 0x3FFFFFF; // bits 0–29

	pub fn new(block_index: u32) -> Self {
		Self(block_index & Self::INDEX_MASK)
	}
	pub fn block_index(&self) -> u32 {
		self.0 & Self::INDEX_MASK
	}
	pub fn is_claimed(&self) -> bool {
		(self.0 & Self::CLAIM_BIT) != 0
	}
	pub fn is_done(&self) -> bool {
		(self.0 & Self::DONE_BIT) != 0
	}
	pub fn claim(&mut self) {
		self.0 |= Self::CLAIM_BIT;
	}
	pub fn mark_done(&mut self) {
		self.0 |= Self::DONE_BIT;
	}
	pub fn reset(&mut self) {
		self.0 = 0;
	}
}