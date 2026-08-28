#[repr(u8)]
pub enum DataFlags {
	UncompressedFlag = 0xFF,
	SignInverseBit = 0x80,
}

impl DataFlags {
	/// Returns the raw byte value of the flag
	#[inline(always)]
	pub fn mask(self) -> u8 {
		self as u8
	}

	/// Returns the bitwise inverted mask byte
	#[inline(always)]
	pub fn inv_mask(self) -> u8 {
		!(self as u8)
	}
}
