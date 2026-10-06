//! One handler per subcommand, grouped by area. Each module holds its subcommands' arguments
//! (the `--help` text of their options) and a function per subcommand that runs it.

pub mod advisor;
pub mod label;
pub mod play;
