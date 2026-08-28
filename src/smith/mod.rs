// The Agent Smith — forge agent configurations by assembling components.
//
// Like a blacksmith: pick Ingots from the Rack, shape them on the Anvil,
// and export a runnable Blueprint.

pub mod anvil;
pub mod blueprint;
pub mod rack;

pub use anvil::Anvil;
pub use blueprint::{Blueprint, BlueprintSlot};
pub use rack::{Ingot, Rack};
