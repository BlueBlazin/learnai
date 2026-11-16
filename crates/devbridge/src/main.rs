//! Dev-only HTTP bridge: serves learnai-core on `127.0.0.1:1421`.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use learnai_core::{Core, CoreOptions, ProgramHarness, RootInputs, SshOptions, resolve_root};
use learnai_devbridge::{DEFAULT_ORIGINS, DEFAULT_PORT, serve};
use tokio::net::TcpListener;

const USAGE: &str = "\
learnai-devbridge: expose learnai-core over HTTP for browser development

Usage: learnai-devbridge [--root <dir>] [--port <port>] [--origin <url>]... [--harness <program>]
                          [--ssh <program>] [--rsync <program>] [--ssh-config <file>]

Options:
  --root <dir>         Content root holding python/, problems/, workspace/
                       (default: $LEARNAI_ROOT, then the repository this was built from)
  --port <port>        Port on 127.0.0.1 (default: 1421)
  --origin <url>       Extra origin allowed by CORS (default: the Vite dev server on 1420)
  --harness <program>  Run <program> with the harness arguments instead of
                       `uv run ... python -m learnai.harness` (for fake harnesses)
  --ssh <program>      Use <program> instead of `ssh` for remote targets (for fake hosts)
  --rsync <program>    Use <program> instead of `rsync` for remote targets
  --ssh-config <file>  Read host aliases from <file> instead of ~/.ssh/config
  -h, --help           Show this help
";

struct Args {
    root: Option<PathBuf>,
    port: u16,
    origins: Vec<String>,
    harness: Option<PathBuf>,
    ssh: SshOptions,
}

fn parse_args(mut argv: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let mut args = Args {
        root: None,
        port: DEFAULT_PORT,
        origins: DEFAULT_ORIGINS
            .iter()
            .map(|origin| (*origin).to_owned())
            .collect(),
        harness: None,
        ssh: SshOptions::default(),
    };
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--root" => args.root = Some(PathBuf::from(value()?)),
            "--port" => {
                let text = value()?;
                args.port = text.parse().map_err(|_| format!("invalid port `{text}`"))?;
            }
            "--origin" => args.origins.push(value()?),
            "--harness" => args.harness = Some(PathBuf::from(value()?)),
            "--ssh" => args.ssh.ssh_program = Some(PathBuf::from(value()?)),
            "--rsync" => args.ssh.rsync_program = Some(PathBuf::from(value()?)),
            "--ssh-config" => args.ssh.config = Some(PathBuf::from(value()?)),
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Some(args))
}

async fn run(args: Args) -> Result<(), String> {
    let root = resolve_root(&RootInputs {
        explicit: args.root,
        ..RootInputs::default()
    })
    .map_err(|e| e.to_string())?;
    let mut options = CoreOptions::new(&root.path);
    options.ssh = args.ssh;
    if let Some(program) = args.harness {
        options.harness = Some(Arc::new(ProgramHarness::new(program)));
    }
    let core = Core::open(options).map_err(|e| e.to_string())?;
    let helpers = core.clone();

    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, args.port));
    let listener = TcpListener::bind(address)
        .await
        .map_err(|e| format!("cannot listen on {address}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(root = %root.path.display(), source = ?root.source, "content root");
    tracing::info!("listening on http://{bound}");
    let served = tokio::select! {
        served = serve(listener, core, &args.origins) => served.map_err(|e| e.to_string()),
        () = stop_signal() => {
            tracing::info!("stopping");
            Ok(())
        }
    };
    // The language server and the docs helper must not outlive the bridge.
    helpers.shutdown();
    served
}

/// Resolves when the process is asked to stop: Ctrl-C, or SIGTERM (what `pnpm dev:real`
/// sends when Vite exits).
async fn stop_signal() {
    let interrupt = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = interrupt => {}
                    _ = terminate.recv() => {}
                }
            }
            Err(_) => {
                let _ = interrupt.await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = interrupt.await;
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
