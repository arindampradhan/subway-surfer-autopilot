//! Checks the human key logger end to end: opens the game, presses keys through CDP (which the
//! page sees as real key events) and prints what `drain_keys` returns.

use std::time::Duration;

use ssbot::browser::Game;
use ssbot::config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::load(std::path::Path::new("ssbot.toml"))?;
    let mut game = Game::launch(&cfg.browser).await?;
    game.watch_keys().await?;
    game.open(&cfg.browser).await?;
    game.focus().await?;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let t0 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64() * 1000.0;
    for k in ["ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown"] {
        game.press(k, Duration::from_millis(60)).await?;
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    let keys = game.drain_keys().await?;
    for k in &keys {
        println!("{:10} at +{:.0} ms", k.key, k.ts - t0);
    }
    println!("{} keys captured (expected 4)", keys.len());
    game.close().await;
    Ok(())
}
