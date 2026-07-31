"""Pure mechanism layer — stateless signal processing operations.

These functions implement textbook DSP and coding theory.
They never change. They have no opinions about when or why to use them.
The policy layer (encode.py, decode.py, etc.) decides what to call.

Each module is a pure function: input → output, no side effects.
"""
