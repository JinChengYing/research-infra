# Research Infrastructure

Long-term research engineering base.

Rules:
- core/ contains stable reusable components
- configs/ define experiment-level variables
- experiments/ may be messy but must not pollute core
- scripts/ automate running experiments

If something can be expressed in configs, it must not be hard-coded.
