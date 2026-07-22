# Contributing

Each file-changing commit records material authorship in Git trailers. Include
one `AI-Assisted-By` JSON trailer per AI contributor and exactly one
`File-Contribution` JSON trailer per changed path. Human contributors use a
`Contributor` JSON trailer and are listed by actor ID in each affected file.

Source comments explain current behavior and rationale; they do not serve as an
edit history. Cryptographic commit signing remains separate from authorship
trailers.
