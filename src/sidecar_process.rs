//! OpenJev inference sidecar: a Python child process speaking JSON lines (SPEC §4.6).
//! Requests and replies go over channels, so the advisor never blocks on the process, and
//! a crashed sidecar just closes the reply channel (the bot then plays reflex-only).

use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::config::AdvisorConfig;
use crate::sidecar::{Reply, Request, SidecarLink};

/// The running process; its link goes to the advisor.
pub struct Sidecar {
    pub model: String,
    child: Child,
}

impl Sidecar {
    /// Starts the sidecar and waits for its `ready` line (model load takes ~1 s on MLX).
    pub async fn spawn(cfg: &AdvisorConfig) -> Result<(Sidecar, SidecarLink)> {
        let mut cmd = Command::new(&cfg.python);
        cmd.arg(&cfg.script).args(["--model", &cfg.model, "--backend", &cfg.backend]);
        if !cfg.head.is_empty() {
            cmd.args(["--head", &cfg.head]);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting sidecar: {} {}", cfg.python, cfg.script))?;
        let mut stdin = child.stdin.take().context("sidecar stdin")?;
        let mut lines = BufReader::new(child.stdout.take().context("sidecar stdout")?).lines();

        let ready = tokio::time::timeout(Duration::from_secs(180), async {
            while let Some(line) = lines.next_line().await? {
                match serde_json::from_str::<Reply>(&line) {
                    Ok(Reply::Ready { model, backend }) => return Ok(format!("{model} ({backend})")),
                    Ok(other) => tracing::debug!("sidecar before ready: {other:?}"),
                    Err(_) => tracing::debug!("sidecar: {line}"),
                }
            }
            bail!("sidecar exited before it was ready")
        })
        .await
        .context("sidecar didn't become ready within 180 s")??;

        let (req_tx, mut req_rx) = mpsc::unbounded_channel::<Request>();
        let (rep_tx, rep_rx) = mpsc::unbounded_channel::<Reply>();
        tokio::spawn(async move {
            while let Some(req) = req_rx.recv().await {
                let mut line = serde_json::to_string(&req).expect("requests serialise");
                line.push('\n');
                if let Err(err) = stdin.write_all(line.as_bytes()).await {
                    tracing::error!("sidecar write failed: {err}; advisor off");
                    break;
                }
                let _ = stdin.flush().await;
            }
        });
        tokio::spawn(async move {
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => match serde_json::from_str::<Reply>(&line) {
                        Ok(reply) => {
                            if rep_tx.send(reply).is_err() {
                                break;
                            }
                        }
                        Err(err) => tracing::warn!("sidecar sent a bad line ({err}): {line}"),
                    },
                    Ok(None) | Err(_) => {
                        tracing::warn!("sidecar output closed (exited or shut down); advisor off");
                        break;
                    }
                }
            }
        });
        Ok((Sidecar { model: ready, child }, SidecarLink { requests: req_tx, replies: rep_rx }))
    }

    pub async fn shutdown(mut self) {
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        let _ = self.child.kill().await;
    }
}
