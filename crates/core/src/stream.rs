//! Reading a child process's output as lines, shared by everything that streams a
//! subprocess: runs, environment installs, and model asset fetches.

use std::io::{BufRead, BufReader, Read};
use std::sync::mpsc::Sender;
use std::thread;

/// What a streaming job's event loop receives: output lines from the reader threads,
/// the end of a pipe, or a request to stop.
pub(crate) enum Message {
    /// One line of standard output.
    Stdout(String),
    /// One line of standard error.
    Stderr(String),
    /// A pipe reached end of input.
    Closed,
    /// The user asked for the job to stop.
    Cancel,
}

/// Forwards each line of `pipe` as a message, then `Closed`. Returns the number of
/// streams now being read (1), or 0 if the reader thread could not be started.
pub(crate) fn spawn_reader(
    pipe: Box<dyn Read + Send>,
    tx: Sender<Message>,
    wrap: fn(String) -> Message,
) -> usize {
    let spawned = thread::Builder::new()
        .name("output-reader".to_owned())
        .spawn(move || {
            for_each_line(pipe, |line| tx.send(wrap(line)).is_ok());
            let _ = tx.send(Message::Closed);
        });
    usize::from(spawned.is_ok())
}

/// Calls `handle` with each line of `reader` (lossy UTF-8, line ending removed) until
/// end of input, a read error, or `handle` returns false.
pub(crate) fn for_each_line(reader: impl Read, mut handle: impl FnMut(String) -> bool) {
    let mut reader = BufReader::new(reader);
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                while matches!(buffer.last(), Some(b'\n' | b'\r')) {
                    buffer.pop();
                }
                if !handle(String::from_utf8_lossy(&buffer).into_owned()) {
                    break;
                }
            }
        }
    }
}
