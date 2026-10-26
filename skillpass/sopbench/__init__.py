"""The SOP-Bench evaluation (Sections 5.3 to 5.5): each domain's standard operating procedure is
a skill, read by a tool-calling agent under `none`, `text`, `plain` and `spass`.

`domain` loads a domain from a checkout of the benchmark; `agent` runs the episodes; `runner`
makes the S-Passes (or a baseline's tensors), runs the readers and writes the evidence;
`report` builds the tables.
"""
