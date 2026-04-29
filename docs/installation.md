# Installation

> **Note:** PyPI distribution is planned. For now, install directly from GitHub.

```bash
pip install git+https://github.com/miniscope/placecell.git
```

## Install oasis-deconv (required for deconvolution)

placecell uses [`oasis-deconv`](https://github.com/j-friedrich/OASIS) for
its deconvolution step. It is **required** for the full pipeline but
**not bundled** with placecell — PyPI wheel coverage is patchy (arm64
macOS only), so we leave the install path to you:

```bash
# recommended: force a source build for consistency across platforms (needs a C compiler)
pip install --no-binary oasis-deconv oasis-deconv

# alternative: prebuilt binaries via conda-forge
conda install -c conda-forge oasis-deconv
```

If the source build fails, that is an upstream issue — see the
[oasis-deconv repository](https://github.com/j-friedrich/OASIS).

## Development

```bash
git clone https://github.com/miniscope/placecell.git
cd placecell
uv sync --extra all
```
