//! Statement figures: files under a problem's `assets/` directory, served as `data:` URLs.
//!
//! The request names a file relative to `assets/`. It is rejected unless every component
//! is a plain name (no `..`, no absolute path, no backslash), the extension is `.svg` or
//! `.png`, the resolved file really lies inside the directory (so a symlink cannot lead
//! out of it), and the file is no larger than [`MAX_ASSET_BYTES`].

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::error::{CoreError, Result};

/// Largest asset that is served. A statement figure is a diagram, not a photograph.
pub const MAX_ASSET_BYTES: u64 = 1024 * 1024;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

/// The kinds of file a statement may embed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssetKind {
    Svg,
    Png,
}

impl AssetKind {
    fn of(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?;
        if extension.eq_ignore_ascii_case("svg") {
            Some(Self::Svg)
        } else if extension.eq_ignore_ascii_case("png") {
            Some(Self::Png)
        } else {
            None
        }
    }

    const fn mime(self) -> &'static str {
        match self {
            Self::Svg => "image/svg+xml",
            Self::Png => "image/png",
        }
    }
}

fn invalid(path: &str, why: &str) -> CoreError {
    CoreError::InvalidArgument(format!("invalid asset path `{path}`: {why}"))
}

/// Checks the shape of a requested asset path and returns it with its kind.
fn checked(path: &str) -> Result<(PathBuf, AssetKind)> {
    if path.is_empty() {
        return Err(invalid(path, "it is empty"));
    }
    if path.contains(['\\', '\0']) {
        return Err(invalid(path, "it contains a backslash or a NUL"));
    }
    let relative = Path::new(path);
    if !relative
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(invalid(
            path,
            "it must stay inside the problem's assets directory",
        ));
    }
    let kind = AssetKind::of(relative)
        .ok_or_else(|| invalid(path, "only .svg and .png files are served"))?;
    Ok((relative.to_path_buf(), kind))
}

/// Reads `path` (relative to `<problem_dir>/assets`) and returns it as a `data:` URL.
pub fn read_asset(problem_dir: &Path, path: &str) -> Result<String> {
    let (relative, kind) = checked(path)?;
    let assets = problem_dir.join("assets");
    let file = assets.join(relative);

    // Resolve symlinks on both sides, then require containment.
    let missing = |e: std::io::Error, at: &Path| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CoreError::MissingFile(at.to_path_buf())
        } else {
            CoreError::io("resolve", at, e)
        }
    };
    let real_assets = fs::canonicalize(&assets).map_err(|e| missing(e, &file))?;
    let real_file = fs::canonicalize(&file).map_err(|e| missing(e, &file))?;
    if !real_file.starts_with(&real_assets) {
        return Err(invalid(
            path,
            "it must stay inside the problem's assets directory",
        ));
    }

    let meta = fs::metadata(&real_file).map_err(|e| CoreError::io("inspect", &file, e))?;
    if !meta.is_file() {
        return Err(CoreError::MissingFile(file));
    }
    if meta.len() > MAX_ASSET_BYTES {
        return Err(CoreError::AssetTooLarge {
            path: file,
            size: meta.len(),
            limit: MAX_ASSET_BYTES,
        });
    }
    let bytes = fs::read(&real_file).map_err(|e| CoreError::io("read", &file, e))?;
    match kind {
        AssetKind::Svg if std::str::from_utf8(&bytes).is_err() => {
            return Err(invalid(path, "the SVG is not UTF-8 text"));
        }
        AssetKind::Png if !bytes.starts_with(PNG_SIGNATURE) => {
            return Err(invalid(path, "the file is not a PNG image"));
        }
        _ => {}
    }
    Ok(format!("data:{};base64,{}", kind.mime(), base64(&bytes)))
}

/// Standard base64 (RFC 4648) with padding.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let sextet = |group: u32, shift: u32| char::from(ALPHABET[((group >> shift) & 0x3f) as usize]);
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = chunk.iter().enumerate().fold(0u32, |acc, (i, &byte)| {
            acc | u32::from(byte) << (16 - 8 * i)
        });
        out.push(sextet(group, 18));
        out.push(sextet(group, 12));
        out.push(if chunk.len() > 1 {
            sextet(group, 6)
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            sextet(group, 0)
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_vectors() {
        let cases = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (input, expected) in cases {
            assert_eq!(base64(input.as_bytes()), expected);
        }
        assert_eq!(base64(&[0xfb, 0xff, 0xfe]), "+//+");
    }

    #[test]
    fn path_shape_is_checked_before_the_file_system() {
        for bad in [
            "",
            "../statement.md",
            "a/../../x.svg",
            "/etc/passwd.svg",
            "./x.svg",
            "a\\..\\x.svg",
            "x.svg\0.png",
            "notes.txt",
            "figure",
            "x.svgz",
            ".svg",
        ] {
            assert!(
                matches!(checked(bad), Err(CoreError::InvalidArgument(_))),
                "{bad:?} should be rejected"
            );
        }
        assert_eq!(
            checked("sub/Figure.SVG").map(|(_, kind)| kind).ok(),
            Some(AssetKind::Svg)
        );
        assert_eq!(
            checked("x.png").map(|(_, kind)| kind).ok(),
            Some(AssetKind::Png)
        );
    }
}
