#[derive(Clone, Copy, Debug)]
pub enum DataFlag {
	GapFlag = 0xFD,
	TailFlag = 0xFE,
	CountFlag = 0xFF,
}


